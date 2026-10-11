"""Renders the diagrams of Entity_Resolution_Diagrams.ipynb:
workflow.png (end-to-end workflow) and pair_decision.png (decision policy for one pair, decide() in step 6).
Run: python entity_flows.py  (needs only matplotlib). Coordinates: x 0..100, y 0..H top to bottom."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

BOX, DIAMOND, OUT, NOTE = '#eef3fa', '#fff1d6', '#e3f1e3', '#f6f6f6'
EDGE, TEXT = '#5b6b80', '#1f2933'
HERE = Path(__file__).parent


def canvas(height, figsize):
    """Start a new figure; the drawing helpers below draw on it."""
    global fig, ax, nodes
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100), ax.set_ylim(height, 0), ax.axis('off')
    nodes = {}


def box(key, x, y, w, h, text, color=BOX, bold_first=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.4,rounding_size=1.2',
                                fc=color, ec=EDGE, lw=1.2))
    lines = text.split('\n')
    if bold_first and len(lines) == 1:
        ax.text(x, y, text, ha='center', va='center', fontsize=10.5, weight='bold', color=TEXT)
    elif bold_first:
        ax.text(x, y - h / 2 + 2.0, lines[0], ha='center', va='center', fontsize=10.5, weight='bold', color=TEXT)
        if len(lines) > 1:
            ax.text(x, y + 1.4, '\n'.join(lines[1:]), ha='center', va='center', fontsize=9, color=TEXT, linespacing=1.35)
    else:
        ax.text(x, y, text, ha='center', va='center', fontsize=9.5, color=TEXT, linespacing=1.35)
    nodes[key] = (x, y, w, h)


def diamond(key, x, y, w, h, text):
    ax.add_patch(Polygon([(x, y - h / 2), (x + w / 2, y), (x, y + h / 2), (x - w / 2, y)], fc=DIAMOND, ec=EDGE, lw=1.2))
    ax.text(x, y, text, ha='center', va='center', fontsize=9.5, color=TEXT, linespacing=1.3)
    nodes[key] = (x, y, w, h)


def port(key, side):
    if isinstance(side, tuple):  # an explicit point on the node's border
        return side
    x, y, w, h = nodes[key]
    return {'top': (x, y - h / 2), 'bottom': (x, y + h / 2), 'left': (x - w / 2, y), 'right': (x + w / 2, y)}[side]


def note(x, y, text, **kw):
    ax.text(x, y, text, fontsize=8.5, color='#3d4a5c', style='italic', **kw)


def arrow(a, b, sa='bottom', sb='top', label=None, via=(), dashed=False, lpos=0.5, ldx=1.0):
    pts = [port(a, sa), *via, port(b, sb)]
    ls = (0, (4, 3)) if dashed else '-'
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=EDGE, lw=1.2, ls=ls)
    ax.annotate('', xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.2, ls=ls, shrinkA=0, shrinkB=1.5))
    if label:
        p, q = pts[-2], pts[-1]
        ax.text(p[0] + (q[0] - p[0]) * lpos + ldx, p[1] + (q[1] - p[1]) * lpos, label, fontsize=8.5,
                color='#3d4a5c', style='italic', ha='left', va='center')


# ==== Diagram 1: end-to-end workflow ====
canvas(217, (13, 26))

# ---- steps 1-3: from NER mentions to linked forms ----
box('in', 50, 5, 46, 6, 'Inputs\nNER mentions (entities_geocoded.parquet) + geocodes.parquet')
box('forms', 50, 16, 50, 7, '1 Mentions → surface forms\none form per TYPE::word; normalized name, transliteration key')
box('gold', 50, 28, 50, 7, '2 Gold labels (optional)\ndev/test split grouped by true identity; hard negatives')
box('thr', 87, 28, 22, 13, 'Thresholds\nlink, margins, context,\nalias rule, promotion:\ntuned on dev split, else\ncalibration.json, else\ndocumented defaults', NOTE)
box('emb', 50, 40, 50, 7, '3a Cohere embeddings\nform name / title / context vectors + every mention sentence')
box('ret', 50, 52, 54, 7, '3b–c Retrieve link candidates\nWikipedia title/redirect, Wikidata search + facts, OSM wikidata tag (LOC)')
box('link', 50, 65, 56, 10, '3d Link forms and mentions\nforms: main model + independent model (no source features)\n'
    'each mention: independent model on its own sentence\n→ VERIFIED / PROBABLE / NONE per form')
box('amb', 50, 79, 56, 9, 'Flag ambiguity per form\n(a) candidate competition  (b) context clusters (blocks only if dev-precise)\n'
    '(c) mentions linked to different QIDs  (d) title link ≠ OSM tag')
diamond('split?', 50, 92, 54, 10, 'Mentions linked to ≥ 2 different QIDs (c)\n(or human mention labels)?')
box('split', 25, 106, 38, 11, 'Split into senses\none sense per QID; unresolved mentions\n→ #unresolved placeholder sense\n(never paired; becomes an unresolved: entity)')
box('keep', 75, 106, 38, 11, 'One sense per form\nflags (a)(b)(d) kept: the form is\nnot VERIFIED and never auto-merges')

# ---- steps 4-6: pairs and decisions ----
box('cand', 50, 120, 56, 7, '4 Candidate sense pairs (same type, placeholders excluded)\nname, rules, fuzz, embeddings, Wikidata, OSM — proposes, never decides')
box('feat', 50, 131, 56, 7, '5 Pair features and hard constraints\nVERIFIED QIDs differ · VERIFIED P131/P361/P749 relation · human cannot-link')
diamond('dec', 50, 143, 40, 9, '6 Decision policy\n(precedence: Diagram 2)')
box('nomerge', 10, 157, 18, 8, 'AUTO_NO_MERGE\nhard constraint', OUT)
box('nonbind', 33, 157, 22, 8, 'LIKELY_DIFFERENT /\nINSUFFICIENT_EVIDENCE\n(non-binding)', OUT, bold_first=False)
box('review', 59, 157, 24, 8, 'REVIEW\nweak evidence, or strong\nwith soft counter-evidence', OUT)
box('merge', 88, 157, 22, 8, 'AUTO_MERGE\nhuman same · strong ·\ncorroborated short form', OUT)
diamond('promo', 59, 171, 30, 10, 'Promoted? dev model score ≥\nthreshold, no soft conflict,\nno ambiguous side')

# ---- steps 7-13: clustering, entities, outputs ----
box('uf', 88, 172, 22, 12, '7 Union-Find\nstrongest edge first:\nhuman > verified QID / alias\n> exact name > short form\n> promoted')
diamond('cons', 80, 188, 30, 10, 'Different VERIFIED QIDs,\nor a cannot-link between\nthe two clusters?')
box('notm', 28, 186, 38, 11, 'Not merged by this pair\nREVIEW pairs also → review_queue.csv;\nthe senses can still join through\nother AUTO_MERGE edges')
box('refuse', 62, 201, 22, 8, 'Refused\nlogged in merge_audit.csv;\nhuman same → adjudication.csv', OUT)
box('union', 89, 201, 16, 7, 'Union\nlogged in merge_audit.csv', OUT)
box('ent', 50, 212, 88, 7, '7–9 Cluster diagnostics → entities (wd:QID / loc: / unresolved:) → graph + JSON export\n'
    'graph = co-occurrence (undirected) + Wikidata relations (directed)  ·  then steps 10–13: test-split evaluation, ablations, review queue, tests')

for a, b in [('in', 'forms'), ('forms', 'gold'), ('gold', 'emb'), ('emb', 'ret'), ('ret', 'link'), ('link', 'amb'),
             ('amb', 'split?'), ('cand', 'feat'), ('feat', 'dec'), ('nonbind', 'notm'), ('review', 'promo'), ('merge', 'uf'),
             ]:
    arrow(a, b)
arrow('gold', 'thr', 'right', 'left')
arrow('thr', 'link', 'bottom', 'right', via=[(87, 65)], dashed=True)
ax.text(88, 50, 'used by\nsteps 3, 5, 6', fontsize=8.5, color='#3d4a5c', style='italic', va='center')
arrow('split?', 'split', 'left', 'top', label='yes', via=[(25, 92)], lpos=0.3)
arrow('split?', 'keep', 'right', 'top', label='no', via=[(75, 92)], lpos=0.3)
arrow('split', 'cand', 'bottom', 'top', via=[(25, 113.75), (50, 113.75)])
arrow('keep', 'cand', 'bottom', 'top', via=[(75, 113.75), (50, 113.75)])
for k in ('nomerge', 'nonbind', 'review', 'merge'):
    arrow('dec', k, 'bottom', 'top', via=[(50, 150), (nodes[k][0], 150)])
arrow('promo', 'uf', 'right', (77, 171))
ax.text(76.5, 176, 'yes → AUTO_MERGE', fontsize=8.5, color='#3d4a5c', style='italic', ha='right')
arrow('promo', 'notm', 'bottom', 'right', via=[(59, 186)])
ax.text(60, 181, 'no', fontsize=8.5, color='#3d4a5c', style='italic')
arrow('uf', 'cons', 'bottom', 'top', via=[(88, 180), (80, 180)])
arrow('cons', 'refuse', 'left', 'top', via=[(62, 188)])
ax.text(63, 192.5, 'yes', fontsize=8.5, color='#3d4a5c', style='italic')
arrow('cons', 'union', 'right', 'top', via=[(89, 188)])
ax.text(90, 192.5, 'no', fontsize=8.5, color='#3d4a5c', style='italic')
for k in ('notm', 'refuse', 'union'):
    arrow(k, 'ent', 'bottom', (nodes[k][0], 208.5))
arrow('nomerge', 'cons', 'left', 'bottom', via=[(1, 157), (1, 195), (80, 195)], dashed=True)
ax.text(3, 193.4, 'cannot-link: checked against every member of both clusters', fontsize=8, color='#3d4a5c', style='italic')
ax.text(99, 1, 'dashed = constraint / data flow, not pair flow', fontsize=8, color='#6b7785', ha='right', va='top')

fig.savefig(HERE / 'workflow.png', dpi=110, bbox_inches='tight', facecolor='white')

# ==== Diagram 2: decision policy for one candidate pair (decide(), step 6), checked top to bottom ====
canvas(158, (13, 19))
box('pair', 31, 4, 38, 5, 'Candidate pair (step 4) with its features (step 5)', bold_first=False)
diamond('hard', 31, 18, 54, 13, 'Hard constraint?\nhuman "different" label · VERIFIED QIDs differ ·\nVERIFIED P131 / P361 / P749 relation')
box('d_nomerge', 80, 18, 28, 7, 'AUTO_NO_MERGE\nbinding cannot-link for step 7', OUT)
diamond('human', 31, 35, 34, 9, 'Human "same" label?')
box('d_human', 80, 35, 28, 6, 'AUTO_MERGE\nhuman must-link', OUT)
diamond('strong', 31, 55, 54, 16, 'Strong evidence?\nsame VERIFIED QID · unambiguous Wikidata\nalias of the other side\'s VERIFIED QID ·\nsame normalized name')
diamond('soft1', 72, 55, 26, 12, 'Any soft\ncounter-evidence?')
box('s_rev', 93, 47, 13, 6, 'REVIEW', OUT)
box('s_mrg', 93, 63, 13, 6, 'AUTO_MERGE', OUT)
diamond('short', 31, 80, 54, 18, 'Corroborated ambiguous short form?\nalias / surname / acronym of the\nother side\'s VERIFIED QID: same documents,\nno rival, mention context prefers it\n(rule enabled on dev)')
diamond('soft2', 72, 80, 26, 13, 'Soft counter-evidence\nother than "probable\nQIDs differ"?')
box('c_rev', 93, 72, 13, 6, 'REVIEW', OUT)
box('c_mrg', 93, 88, 13, 6, 'AUTO_MERGE', OUT)
diamond('weak', 31, 105, 54, 18, 'Weak evidence?\nrule · initials · translit · fuzz ≥ 90 ·\nname embedding ≥ 0.80 ·\nsame probable QID · same OSM object ·\nuncorroborated short form')
diamond('emb', 72, 105, 26, 13, 'Probable QIDs differ\nand the only evidence\nis name embedding?')
box('w_ld', 93, 97, 13, 6, 'LIKELY_\nDIFFERENT', OUT, bold_first=False)
box('w_rev', 93, 113, 13, 6, 'REVIEW', OUT)
diamond('both', 31, 127, 40, 10, 'Both sides have a\nprobable QID?')
box('n_ld', 72, 127, 22, 6, 'LIKELY_DIFFERENT', OUT, bold_first=False)
box('n_ins', 31, 141, 28, 6, 'INSUFFICIENT_EVIDENCE', OUT, bold_first=False)
note(72, 141, 'soft counter-evidence = ambiguous side ·\nprobable QIDs differ · place_rank differs ·\npolygon containment', ha='center', va='center')
box('after', 50, 153, 90, 8, 'After decide() (see Diagram 1)\nREVIEW may be promoted to AUTO_MERGE by the dev-trained model · every AUTO_MERGE is only a proposal,\n'
    're-checked against whole clusters in step 7 · only AUTO_NO_MERGE is binding', NOTE)

for a, b in [('pair', 'hard'), ('hard', 'human'), ('human', 'strong'), ('strong', 'short'), ('short', 'weak'), ('weak', 'both'), ('both', 'n_ins')]:
    arrow(a, b)
for a, b in [('hard', 'd_nomerge'), ('human', 'd_human'), ('strong', 'soft1'), ('short', 'soft2'), ('weak', 'emb'), ('both', 'n_ld')]:
    arrow(a, b, 'right', 'left')
    note(nodes[a][0] + nodes[a][2] / 2 + 0.6, nodes[a][1] - 1.6, 'yes')
for a in ('hard', 'human', 'strong', 'short', 'weak', 'both'):
    note(nodes[a][0] + 1, nodes[a][1] + nodes[a][3] / 2 + 1.4, 'no')
for d, yes, no in [('soft1', 's_rev', 's_mrg'), ('soft2', 'c_rev', 'c_mrg'), ('emb', 'w_ld', 'w_rev')]:
    x, y, w, h = nodes[d]
    arrow(d, yes, 'top', 'left', via=[(x, nodes[yes][1])])
    arrow(d, no, 'bottom', 'left', via=[(x, nodes[no][1])])
    note(x + 1, y - h / 2 - 1.3, 'yes')
    note(x + 1, y + h / 2 + 1.5, 'no')
fig.savefig(HERE / 'pair_decision.png', dpi=110, bbox_inches='tight', facecolor='white')
