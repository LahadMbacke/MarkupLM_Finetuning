"""
Inference MarkupLM finetuné (SWDE) sur données Ideactiv
========================================================
Charge un checkpoint SWDE finetuné et prédit les attributs
pour chaque nœud DOM des événements Ideactiv.

Usage
-----
python3 markuplm_infer_events.py \
    --data_dir    ../../data_by_domain \
    --model_dir   /path/to/results_markuplm/restaurant/seed-1_pages-2000/<site> \
    --max_events  3
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import torch
import lxml.html
from lxml import etree
from lxml.html.clean import Cleaner
from transformers import MarkupLMProcessor

# Label mapping from unilm_repo/markuplm/examples/fine_tuning/run_swde/constants.py
# (ATTRIBUTES_PLUS_NONE — alphabetical order, "none" = background class)
SWDE_LABELS = {
    "auto":       ["engine", "fuel_economy", "model", "none", "price"],
    "book":       ["author", "isbn_13", "none", "publication_date", "publisher", "title"],
    "camera":     ["manufacturer", "model", "none", "price"],
    "job":        ["company", "date_posted", "location", "none", "title"],
    "movie":      ["director", "genre", "mpaa_rating", "none", "title"],
    "nbaplayer":  ["height", "name", "none", "team", "weight"],
    "restaurant": ["address", "cuisine", "name", "none", "phone"],
    "university": ["name", "none", "phone", "type", "website"],
}

# The SWDE training script (unilm_repo) saves models with the custom markuplmft class
# (classification head named token_cls.* instead of classifier.*).
# Use that class when available so weights load correctly.
_markuplmft_path = Path(__file__).resolve().parent / "unilm_repo" / "markuplm"
if _markuplmft_path.exists():
    sys.path.insert(0, str(_markuplmft_path))
    from markuplmft.models.markuplm import MarkupLMForTokenClassification
else:
    from transformers import MarkupLMForTokenClassification


# ── HTML → nœuds DOM ─────────────────────────────────────────────────────────

def clean_spaces(text):
    return " ".join(re.split(r"\s+", text.strip()))


def clean_format_str(text):
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = "".join(c if ord(c) < 128 else "" for c in text)
    return clean_spaces(text)


def get_dom_tree(html):
    cleaner = Cleaner()
    cleaner.javascript = True
    cleaner.style = True
    cleaner.page_structure = False
    html = html.replace("\0", "")
    for tag in ("<br>", "<br/>", "<br />", "<BR>", "<BR/>", "<BR />"):
        html = html.replace(tag, "--BRRB--")
    html = clean_format_str(html)
    root = cleaner.clean_html(lxml.html.fromstring(html))
    return etree.ElementTree(root)


def extract_nodes(dom):
    nodes = []
    for e in dom.iter():
        for content, suffix in [(e.text, ""), (e.tail, "/tail")]:
            if not content:
                continue
            for i, part in enumerate(content.split("--BRRB--")):
                text = clean_spaces(part)
                if not text:
                    continue
                xpath = dom.getpath(e) + suffix
                if len(content.split("--BRRB--")) >= 2:
                    xpath += f"/br[{i + 1}]"
                nodes.append((text, xpath))
    return nodes


# ── Inférence MarkupLM ────────────────────────────────────────────────────────

def run_inference(nodes, processor, model, device, id2label, max_seq_length=384):
    node_texts  = [t for t, _ in nodes]
    node_xpaths = [x for _, x in nodes]

    encoding = processor.tokenizer(
        node_texts,
        xpaths=node_xpaths,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**encoding).logits[0]  # (seq_len, num_labels)

    pred_ids  = logits.argmax(dim=-1).cpu().tolist()
    input_ids = encoding["input_ids"][0].cpu().tolist()
    tokens    = processor.tokenizer.convert_ids_to_tokens(input_ids)

    bg_labels = {"O", "none", "LABEL_0"}
    special   = {"[PAD]", "[CLS]", "[SEP]", "<pad>", "<s>", "</s>"}
    results = []
    for tok, pred_id in zip(tokens, pred_ids):
        label = id2label.get(pred_id, f"LABEL_{pred_id}")
        if label not in bg_labels and tok not in special:
            results.append((tok, label))
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       default="../../data_by_domain",
                        help="Répertoire contenant les .json Ideactiv")
    parser.add_argument("--model_dir",      required=True,
                        help="Checkpoint MarkupLM finetuné (dossier HuggingFace)")
    parser.add_argument("--processor_dir",  default=None,
                        help="Dossier du processor/tokenizer (défaut: model_dir). "
                             "Utiliser le modèle de base si le checkpoint ne contient pas preprocessor_config.json")
    parser.add_argument("--max_events",     type=int, default=3)
    parser.add_argument("--domain",         default=None,
                        help="Filtrer sur un domaine (préfixe du nom de fichier)")
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--vertical",       default=None,
                        help="Vertical SWDE du modèle (ex: restaurant). "
                             "Permet d'afficher les vrais noms d'attributs.")
    args = parser.parse_args()

    processor_dir = args.processor_dir or args.model_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    print(f"Chargement processor : {processor_dir}")
    print(f"Chargement modèle   : {args.model_dir}")
    processor = MarkupLMProcessor.from_pretrained(processor_dir)
    processor.parse_html = False
    model     = MarkupLMForTokenClassification.from_pretrained(args.model_dir)
    model.to(device).eval()

    # Build id2label: use SWDE constants if vertical provided, else fall back to config
    if args.vertical and args.vertical in SWDE_LABELS:
        id2label = {i: lbl for i, lbl in enumerate(SWDE_LABELS[args.vertical])}
    else:
        num_out = model.token_cls.decoder.out_features if hasattr(model, "token_cls") else len(model.config.id2label)
        id2label = {i: model.config.id2label.get(i, f"LABEL_{i}") for i in range(num_out)}
    print(f"Labels du modèle : {id2label}\n")

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.json"))
    if args.domain:
        files = [f for f in files if f.name.startswith(args.domain)]

    count = 0
    for json_file in files:
        if count >= args.max_events:
            break
        try:
            raw = json_file.read_bytes().decode("utf-8", errors="ignore")
            data = json.loads(raw, strict=False)
        except json.JSONDecodeError as e:
            print(f"[ERR JSON] {json_file.name} — {e}")
            continue
        for event in data.get("events", []):
            if count >= args.max_events:
                break
            html = event.get("pageContent", "")
            if not html:
                continue

            try:
                dom   = get_dom_tree(html)
                nodes = extract_nodes(dom)
            except Exception as e:
                print(f"[ERR DOM] {json_file.name} — {e}")
                count += 1
                continue

            if not nodes:
                continue

            fd = event.get("finalData", {})
            print(f"{'='*60}")
            print(f"Événement {count + 1} — {json_file.name}")
            print(f"URL           : {event.get('pageUrl', '?')}")
            print(f"GT title      : {fd.get('title', '?')}")
            print(f"GT placeName  : {fd.get('placeName', '?')}")
            print(f"GT address    : {fd.get('placeAddress', '?')} {fd.get('placeZip', '')} {fd.get('placeCity', '')}")
            print(f"GT tel        : {fd.get('tel', '?')}")
            print(f"Nœuds DOM     : {len(nodes)}")

            try:
                predictions = run_inference(
                    nodes, processor, model, device, id2label, args.max_seq_length
                )
            except Exception as e:
                import traceback
                print(f"[ERR INFER] {type(e).__name__}: {e}")
                traceback.print_exc()
                count += 1
                continue

            print(f"\n── Tokens prédits (non-O) ──")
            if not predictions:
                print("  (aucun)")
            else:
                # Grouper par label pour afficher proprement
                by_label = {}
                for tok, label in predictions:
                    by_label.setdefault(label, []).append(tok)
                for label, toks in sorted(by_label.items()):
                    text = processor.tokenizer.convert_tokens_to_string(toks)
                    print(f"  [{label}] {text}")

            print()
            count += 1


if __name__ == "__main__":
    main()
