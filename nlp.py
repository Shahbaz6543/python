#!/usr/bin/env python3
"""
ner_optimized.py

Optimized single-file NER from-scratch:
 - Data loader (CoNLL-style train/dev/test)
 - Feature extraction (prefix/suffix, shape, context, char ngrams)
 - HMM (MLE + add-k smoothing + Viterbi)
 - Averaged Perceptron sequence tagger (feature -> int indexing + averaging)
 - IOB chunk evaluation (entity-level P/R/F1)
 - Simple Tkinter GUI for demo
 - Optionally saves/loads perceptron feature index & weights (pickle)

Usage:
  python ner_optimized.py --data_dir ./data --model perceptron --epochs 5 --gui

Notes:
 - Ensure train.txt and test.txt exist in --data_dir (dev.txt optional).
 - Files should be CoNLL-style: one token per line with final column the NER tag.
   Blank lines separate sentences.

Author: ChatGPT (adapted for your project)
"""
import os
import argparse
from collections import defaultdict, Counter
import math
import random
import pickle
import numpy as np

# ---------------------- Data loader ----------------------
def read_conll(path):
    sents = []
    sent = []
    with open(path, encoding='utf8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                if sent:
                    sents.append(sent)
                    sent = []
                continue
            parts = line.split()
            word = parts[0]
            tag = parts[-1]
            sent.append((word, tag))
    if sent:
        sents.append(sent)
    return sents

# ---------------------- Feature extraction ----------------------
def word_shape(w):
    s = []
    for ch in w:
        if ch.isupper(): s.append('X')
        elif ch.islower(): s.append('x')
        elif ch.isdigit(): s.append('d')
        else: s.append(ch)
    return ''.join(s)

def extract_feature_strings(words, i, prev_tag=None):
    """Return list of feature strings for token words[i]."""
    w = words[i]
    wl = w.lower()
    feats = []
    feats.append('w=' + wl)
    feats.append('w0=' + w)
    if w.istitle(): feats.append('is_title')
    if w.isupper(): feats.append('is_upper')
    if any(ch.isdigit() for ch in w): feats.append('has_digit')
    feats.append('shape=' + word_shape(w))
    if len(wl) >= 3:
        feats.append('pref3=' + wl[:3])
        feats.append('suf3=' + wl[-3:])
    if len(wl) >= 2:
        feats.append('pref2=' + wl[:2])
        feats.append('suf2=' + wl[-2:])
    if i > 0:
        feats.append('-1=' + words[i-1].lower())
    else:
        feats.append('BOS')
    if i < len(words)-1:
        feats.append('+1=' + words[i+1].lower())
    else:
        feats.append('EOS')
    if prev_tag is not None:
        feats.append('prev_tag=' + prev_tag)
    # short char n-grams to help OOV
    chs = wl
    for k in (2,3):
        if len(chs) >= k:
            for j in range(len(chs)-k+1):
                feats.append(f'c{k}=' + chs[j:j+k])
    return feats

# ---------------------- IOB chunk utilities ----------------------
def iob_chunks(tags):
    chunks = []
    start = None
    typ = None
    for i, t in enumerate(tags):
        if t == 'O':
            if typ is not None:
                chunks.append((typ, start, i))
                typ = None
                start = None
        else:
            if '-' not in t:
                # malformed tag, treat as O
                if typ is not None:
                    chunks.append((typ, start, i))
                    typ = None
                    start = None
                continue
            prefix, label = t.split('-', 1)
            if prefix == 'B' or typ is None or label != typ:
                if typ is not None:
                    chunks.append((typ, start, i))
                typ = label
                start = i
            else:
                # continuation
                pass
    if typ is not None:
        chunks.append((typ, start, len(tags)))
    return chunks

def chunk_f1(true_tags_list, pred_tags_list):
    tp = fp = fn = 0
    for true_tags, pred_tags in zip(true_tags_list, pred_tags_list):
        t_chunks = set(iob_chunks(true_tags))
        p_chunks = set(iob_chunks(pred_tags))
        tp += len(t_chunks & p_chunks)
        fp += len(p_chunks - t_chunks)
        fn += len(t_chunks - p_chunks)
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1, tp, fp, fn

# ---------------------- HMM Tagger ----------------------
class HMMTagger:
    def __init__(self, k=1e-5):
        self.k = k
        self.tags = []
        self.emission = defaultdict(Counter)
        self.transition = defaultdict(Counter)
        self.tag_counts = Counter()
        self.vocab = set()

    def train(self, sents):
        for sent in sents:
            prev = 'BOS'
            for word, tag in sent:
                self.emission[tag][word] += 1
                self.tag_counts[tag] += 1
                self.transition[prev][tag] += 1
                prev = tag
                self.vocab.add(word)
            self.transition[prev]['EOS'] += 1
        self.tags = sorted(list(self.tag_counts.keys()))

    def viterbi(self, words):
        T = len(words)
        N = len(self.tags)
        if T == 0:
            return []
        V = np.full((T, N), -1e12)
        back = np.zeros((T, N), dtype=int)
        # init
        for j, tag in enumerate(self.tags):
            ew = self.emission[tag].get(words[0], 0)
            prob_e = math.log((ew + self.k) / (self.tag_counts[tag] + self.k*(len(self.vocab)+1)))
            tw = self.transition['BOS'].get(tag, 0)
            ttot = sum(self.transition['BOS'].values())
            prob_t = math.log((tw + self.k) / (ttot + self.k*(len(self.tags)+1)))
            V[0,j] = prob_e + prob_t
        for t in range(1, T):
            for j, tag in enumerate(self.tags):
                ew = self.emission[tag].get(words[t], 0)
                prob_e = math.log((ew + self.k) / (self.tag_counts[tag] + self.k*(len(self.vocab)+1)))
                best_score = -1e12
                best_i = 0
                for i, prev_tag in enumerate(self.tags):
                    trans = self.transition[prev_tag].get(tag, 0)
                    ttot = sum(self.transition[prev_tag].values())
                    prob_t = math.log((trans + self.k) / (ttot + self.k*(len(self.tags)+1)))
                    score = V[t-1,i] + prob_t + prob_e
                    if score > best_score:
                        best_score = score
                        best_i = i
                V[t,j] = best_score
                back[t,j] = best_i
        best_score = -1e12
        best_i = 0
        for i, tag in enumerate(self.tags):
            trans = self.transition[tag].get('EOS', 0)
            ttot = sum(self.transition[tag].values())
            prob_t = math.log((trans + self.k) / (ttot + self.k*(len(self.tags)+1)))
            if V[T-1,i] + prob_t > best_score:
                best_score = V[T-1,i] + prob_t
                best_i = i
        pred = [None]*T
        idx = best_i
        for t in range(T-1, -1, -1):
            pred[t] = self.tags[idx]
            idx = back[t, idx]
        return pred

# ---------------------- Perceptron Tagger (optimized) ----------------------
class FeatureIndexer:
    def __init__(self):
        self.f2i = {}
        self.i2f = []
    def get(self, f, add=True):
        if f in self.f2i:
            return self.f2i[f]
        if not add:
            return None
        idx = len(self.i2f)
        self.f2i[f] = idx
        self.i2f.append(f)
        return idx
    def __len__(self):
        return len(self.i2f)

class AveragedPerceptron:
    def __init__(self):
        # weights: fid -> {tag: weight}
        self.weights = defaultdict(lambda: defaultdict(float))
        self.classes = set()
        self._totals = defaultdict(lambda: defaultdict(float))
        self._tstamps = defaultdict(lambda: defaultdict(int))
        self.i = 0

    def predict(self, feats):
        scores = defaultdict(float)
        for fid in feats:
            if fid >= 0:
                wdict = self.weights.get(fid)
                if not wdict: continue
                for tag, w in wdict.items():
                    scores[tag] += w
        if not scores:
            return 'O'
        return max(self.classes, key=lambda t: scores.get(t, 0.0))

    def update(self, truth, guess, feats):
        self.i += 1
        if truth == guess: return
        self.classes.add(truth); self.classes.add(guess)
        for fid in feats:
            # truth
            self._totals[fid][truth] += (self.i - self._tstamps[fid].get(truth, 0)) * self.weights[fid].get(truth, 0)
            self._tstamps[fid][truth] = self.i
            self.weights[fid][truth] += 1.0
            # guess
            self._totals[fid][guess] += (self.i - self._tstamps[fid].get(guess, 0)) * self.weights[fid].get(guess, 0)
            self._tstamps[fid][guess] = self.i
            self.weights[fid][guess] -= 1.0

    def average(self):
        for fid, tagdict in list(self.weights.items()):
            for tag, weight in list(tagdict.items()):
                total = self._totals[fid].get(tag, 0.0) + (self.i - self._tstamps[fid].get(tag, 0)) * weight
                averaged = total / float(self.i) if self.i > 0 else weight
                if abs(averaged) > 1e-12:
                    self.weights[fid][tag] = averaged
                else:
                    del self.weights[fid][tag]
            if not self.weights[fid]:
                del self.weights[fid]

class PerceptronTagger:
    def __init__(self):
        self.indexer = FeatureIndexer()
        self.model = AveragedPerceptron()
        self.tags = set()

    def featurize_sentence(self, words, prev_tag=None, add=True):
        feats = []
        # here prev_tag used only for first token to build features that depend on prev_tag
        for i in range(len(words)):
            pf = extract_feature_strings(words, i, prev_tag if i==0 else None)
            ids = [self.indexer.get(f, add) for f in pf]
            ids = [x for x in ids if x is not None]
            feats.append(ids)
        return feats

    def train(self, sents, epochs=5):
        for epoch in range(epochs):
            random.shuffle(sents)
            for sent in sents:
                words = [w for w,t in sent]
                gold = [t for w,t in sent]
                prev_tag = 'O'
                feats = self.featurize_sentence(words, add=True)
                for i in range(len(words)):
                    fid_list = feats[i]
                    guess = self.model.predict(fid_list)
                    truth = gold[i]
                    if guess != truth:
                        self.model.update(truth, guess, fid_list)
                    prev_tag = guess
                    self.tags.add(truth)
        self.model.average()

    def predict(self, words):
        preds = []
        prev_tag = 'O'
        for i in range(len(words)):
            fs = extract_feature_strings(words, i, prev_tag)
            ids = [self.indexer.f2i.get(f, -1) for f in fs]
            ids = [x for x in ids if x >= 0]
            guess = self.model.predict(ids)
            preds.append(guess)
            prev_tag = guess
        return preds

# ---------------------- Driver ----------------------
def train_and_eval(data_dir, model_name='perceptron', epochs=5, gui=False, save_model=None, load_model=None):
    train_file = os.path.join(data_dir, 'train.txt')
    dev_file = os.path.join(data_dir, 'dev.txt')
    test_file = os.path.join(data_dir, 'test.txt')
    assert os.path.exists(train_file), 'train.txt not found'
    assert os.path.exists(test_file), 'test.txt not found'
    train_sents = read_conll(train_file)
    dev_sents = read_conll(dev_file) if os.path.exists(dev_file) else []
    test_sents = read_conll(test_file)
    print(f'Loaded: train={len(train_sents)} dev={len(dev_sents)} test={len(test_sents)}')

    hmm = None; per = None
    if model_name in ('hmm','both'):
        print('Training HMM...')
        hmm = HMMTagger()
        hmm.train(train_sents)
        print('HMM trained')
    if model_name in ('perceptron','both'):
        per = PerceptronTagger()
        if load_model and os.path.exists(load_model):
            print('Loading perceptron model from', load_model)
            with open(load_model, 'rb') as fh:
                per = pickle.load(fh)
        else:
            print('Training Perceptron...')
            per.train(train_sents, epochs=epochs)
            print('Perceptron trained')
            if save_model:
                print('Saving perceptron model to', save_model)
                with open(save_model, 'wb') as fh:
                    pickle.dump(per, fh)

    def eval_model(model, name):
        all_true = []
        all_pred = []
        for sent in test_sents:
            words = [w for w,t in sent]
            true = [t for w,t in sent]
            pred = model.viterbi(words) if isinstance(model, HMMTagger) else model.predict(words)
            all_true.append(true)
            all_pred.append(pred)
        prec, rec, f1, tp, fp, fn = chunk_f1(all_true, all_pred)
        print(f'{name}: P={prec:.4f} R={rec:.4f} F1={f1:.4f} (tp={tp} fp={fp} fn={fn})')
        return all_true, all_pred

    results = {}
    if hmm:
        results['hmm'] = eval_model(hmm, 'HMM')
    if per:
        results['perceptron'] = eval_model(per, 'Perceptron')

    # Error analysis (showing top confusions + sample mismatches)
    for name, (true_list, pred_list) in results.items():
        print('\nError analysis for', name)
        conf = Counter()
        examples = []
        for true_tags, pred_tags, sent in zip(true_list, pred_list, test_sents):
            if true_tags != pred_tags:
                examples.append((sent, true_tags, pred_tags))
                for t,p in zip(true_tags, pred_tags):
                    if t != p:
                        conf[(t,p)] += 1
        print('Top confusions:')
        for (t,p),c in conf.most_common(10):
            print(f'  {t} -> {p}: {c}')
        print('\nSample mismatches (up to 5):')
        for sent, ttags, ptags in examples[:5]:
            words = [w for w,_ in sent]
            print(' '.join(words))
            print('TRUE:', ' '.join(ttags))
            print('PRED:', ' '.join(ptags))
            print('---')

    if gui:
        run_gui(hmm, per)

# ---------------------- GUI ----------------------
def run_gui(hmm, per):
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except Exception as e:
        print('Tkinter not available:', e)
        return
    root = tk.Tk(); root.title('NER Demo')
    tk.Label(root, text='Enter sentence:').pack()
    entry = tk.Entry(root, width=100); entry.pack()
    out = scrolledtext.ScrolledText(root, width=100, height=20); out.pack()
    def pred():
        sent = entry.get().strip()
        if not sent: return
        words = sent.split()
        out.delete('1.0', 'end')
        if hmm:
            ph = hmm.viterbi(words)
            out.insert('end', 'HMM:\n' + ' '.join(f'{w}/{t}' for w,t in zip(words, ph)) + '\n\n')
        if per:
            pp = per.predict(words)
            out.insert('end', 'Perceptron:\n' + ' '.join(f'{w}/{t}' for w,t in zip(words, pp)) + '\n')
    tk.Button(root, text='Predict', command=pred).pack()
    root.mainloop()

# ---------------------- CLI ----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='directory with train.txt dev.txt test.txt')
    parser.add_argument('--model', choices=['hmm','perceptron','both'], default='perceptron')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--save_model', type=str, default=None, help='path to save perceptron (pickle)')
    parser.add_argument('--load_model', type=str, default=None, help='path to load perceptron (pickle)')
    args = parser.parse_args()
    train_and_eval(args.data_dir, model_name=args.model, epochs=args.epochs, gui=args.gui,
                   save_model=args.save_model, load_model=args.load_model)
