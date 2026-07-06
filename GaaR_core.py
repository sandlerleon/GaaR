#!/usr/bin/env python3
"""GaaR_core — compact Goal-Aware Adaptive Regulation framework (Leon Sandler).
Evolutionary search on consumer hardware, one discipline-independent problem:
"a bounded genome, a budgeted fitness oracle, and a provenance log — find the best
phenotype the device can afford." Reusable across vision NAS, language-model NAS,
physics parameter landscapes, and production pipelines.

Invoke by name across chats:  "use GaaR to search <genome> for <objective>"
One-call API:  from GaaR_core import GaaR, GridDomain
               GaaR(domain, strategy="greedy").run(budget=120)
CLI:           python GaaR_core.py --demo            # self-test of every strategy
               python GaaR_core.py --myuncle         # search MyUncle 6000-grid (needs cache)
Spec + measured results: GaaR_MASTER_REFERENCE.md. v2, July 2026.

v2 changes (evidence-based, see MASTER ref):
  * exhaustion escape: greedy mutation around a corner incumbent reaches as few as
    2^d unique mutants; v1 could deadlock/starve there. v2 detects a duplicate-draw
    streak and jumps to a random unseen genome. (Found by testing on the MyUncle grid.)
  * strategy is selectable: greedy (default — measured best), restart, population
    (regularized-evolution-style aging; better on deceptive landscapes), random (baseline).
  * domains are pluggable adapters; SQLite schema is domain-agnostic (genome as JSON).
Needs numpy. Torch domains (CNN/LLM) need torch (+torch-directml on Windows/AMD)."""
from __future__ import annotations
import argparse, json, math, os, random, sqlite3, time
from dataclasses import dataclass, field

# ---------- device (consumer-hardware first) ----------
def pick_device():
    try:
        import torch_directml; return torch_directml.device(), "directml"
    except Exception: pass
    try:
        import torch
        if torch.cuda.is_available(): return torch.device("cuda"), "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"
    except Exception:
        return None, "numpy-only"

# ---------- domain adapter protocol ----------
@dataclass
class Domain:
    """A search domain = genome space + mutation + fitness oracle.
    seed_genome() -> dict; mutate(genome, rng) -> dict;
    evaluate(genome) -> (score: float, info: dict)   (higher is better)"""
    name: str = "abstract"
    def seed_genome(self): raise NotImplementedError
    def mutate(self, genome, rng): raise NotImplementedError
    def evaluate(self, genome): raise NotImplementedError

class GridDomain(Domain):
    """Discrete parameter grid with a cached/aligned fitness array or callable.
    grid: {gene: [values...]}; fitness: array over the itertools.product order, or f(dict)->float."""
    def __init__(self, grid, fitness, name="grid"):
        import itertools
        self.name = name; self.grid = grid; self.keys = list(grid)
        self.sizes = [len(grid[k]) for k in self.keys]
        self._fit = fitness
        self._combos = None
        if not callable(fitness):
            self._combos = list(itertools.product(*[grid[k] for k in self.keys]))
    def _flat(self, ix):
        f = 0
        for i, s in zip(ix, self.sizes): f = f * s + i
        return f
    def _ix_of(self, genome): return tuple(self.grid[k].index(genome[k]) for k in self.keys)
    def seed_genome(self):
        rng = random.Random()
        return {k: rng.choice(v) for k, v in self.grid.items()}
    def mutate(self, genome, rng, per_gene=0.35):
        ix = list(self._ix_of(genome))
        for d in range(len(ix)):
            if rng.random() < per_gene:
                ix[d] = max(0, min(self.sizes[d] - 1, ix[d] + rng.choice([-1, 1])))
        return {k: self.grid[k][i] for k, i in zip(self.keys, ix)}
    def evaluate(self, genome):
        if callable(self._fit): return float(self._fit(genome)), {}
        return float(self._fit[self._flat(self._ix_of(genome))]), {}

class TransformerLMDomain(Domain):
    """Micro-LLM NAS on DirectML/CUDA/CPU — phenotype from train-llm-from-scratch.
    Fitness: -val_loss after `steps` optimizer steps on char-level tiny-shakespeare.
    (Measured on RX 5700: 8 gens x 300 steps in 3m16s; champion 0.8M params ppl 7.5.)"""
    GENES = {"n_embed": (64, 384, 64), "n_blocks": (1, 6, 1)}
    HEADS = [2, 4, 8]; CTX = [64, 128, 192]; BS = [8, 16, 32]
    def __init__(self, repo_path, data_path, steps=300, name="transformer_lm"):
        self.name = name; self.repo = repo_path; self.data = data_path; self.steps = steps
        self.device, self.backend = pick_device()
    def seed_genome(self):
        return dict(n_embed=128, n_head=4, n_blocks=2, context_length=128,
                    batch_size=16, learning_rate=3e-3)
    def mutate(self, g, rng):
        m = dict(g)
        if rng.random() < 0.9:
            m["learning_rate"] = max(1e-4, min(1e-2, m["learning_rate"] * rng.choice([0.7, 1.4])))
            m["batch_size"] = rng.choice(self.BS)
            m["n_embed"] = max(64, min(384, m["n_embed"] + rng.choice([-64, 64])))
            m["n_head"] = rng.choice(self.HEADS)
            m["n_blocks"] = max(1, min(6, m["n_blocks"] + rng.choice([-1, 1])))
            m["context_length"] = rng.choice(self.CTX)
        m["n_embed"] = max(m["n_head"], (m["n_embed"] // m["n_head"]) * m["n_head"])
        return m
    def evaluate(self, g):
        import sys, torch
        sys.path.insert(0, self.repo)
        from src.models.transformer import Transformer
        text = open(self.data, encoding="utf-8").read()
        chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        split = int(0.9 * len(data)); train, val = data[:split], data[split:]
        def batch(src):
            import torch as t
            ix = t.randint(len(src) - g["context_length"] - 1, (g["batch_size"],))
            x = t.stack([src[i:i + g["context_length"]] for i in ix])
            y = t.stack([src[i + 1:i + 1 + g["context_length"]] for i in ix])
            return x.to(self.device), y.to(self.device)
        model = Transformer(n_head=g["n_head"], n_embed=g["n_embed"],
                            context_length=g["context_length"], vocab_size=len(chars),
                            N_BLOCKS=g["n_blocks"]).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=g["learning_rate"])
        model.train()
        for _ in range(self.steps):
            x, y = batch(train); _, loss = model(x, y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = sum(model(*batch(val))[1].item() for _ in range(10)) / 10
        n = sum(p.numel() for p in model.parameters())
        return -v, dict(val_loss=v, ppl=math.exp(min(v, 20)), n_params=n)

# ---------- the regulation engine ----------
@dataclass
class GaaR:
    domain: Domain
    strategy: str = "greedy"           # greedy | restart | population | random
    db_path: str = "gaar_experiments.db"
    pop_size: int = 8
    restart_after: int = 15
    escape_after: int = 40             # v2: duplicate-draw streak -> random unseen jump
    seed: int = 0
    log: bool = True

    def _db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, domain TEXT,
            strategy TEXT, approach TEXT, genome TEXT, score REAL, info TEXT,
            generation INT)""")
        conn.commit(); return conn

    def run(self, budget=120, verbose=True):
        rng = random.Random(self.seed)
        conn = self._db() if self.log else None
        seen = {}; pop = []; curve = []
        key = lambda g: json.dumps(g, sort_keys=True)
        def evaluate(g):
            k = key(g)
            if k in seen: return seen[k], False
            s, info = self.domain.evaluate(g); seen[k] = (s, info)
            return (s, info), True
        best_g = self.domain.seed_genome()
        (best, binfo), _ = evaluate(best_g); curve.append(best)
        stall = dup = gen = 0
        if conn:
            conn.execute("INSERT INTO experiments (timestamp,domain,strategy,approach,genome,score,info,generation) VALUES (?,?,?,?,?,?,?,?)",
                         (time.strftime("%Y-%m-%dT%H:%M:%S"), self.domain.name, self.strategy,
                          "seed", key(best_g), best, json.dumps(binfo), 0)); conn.commit()
        while len(seen) < budget:
            gen += 1
            if dup > self.escape_after:                      # v2 exhaustion escape
                g = self.domain.seed_genome()
                while key(g) in seen: g = self.domain.seed_genome()
                approach = "escape_random"; dup = 0
            elif self.strategy == "random":
                g = self.domain.seed_genome(); approach = "random"
            elif self.strategy in ("greedy", "restart"):
                parent = best_g
                if self.strategy == "restart" and stall >= self.restart_after:
                    parent = self.domain.seed_genome(); stall = 0
                g = self.domain.mutate(parent, rng); approach = "mutate_best"
            elif self.strategy == "population":
                if len(pop) < self.pop_size:
                    g = self.domain.seed_genome(); approach = "populate"
                else:
                    a, b = rng.randrange(len(pop)), rng.randrange(len(pop))
                    parent = pop[a] if pop[a][1] >= pop[b][1] else pop[b]
                    g = self.domain.mutate(parent[0], rng); approach = "tournament_mutate"
            pre = len(seen); (s, info), fresh = evaluate(g)
            dup = 0 if fresh else dup + 1
            if not fresh: continue
            if self.strategy == "population":
                pop.append((g, s));  pop.pop(0) if len(pop) > self.pop_size else None
            if s > best: best, best_g, stall, approach = s, g, 0, "exploit_" + approach
            else: stall += 1
            curve.append(best)
            if conn:
                conn.execute("INSERT INTO experiments (timestamp,domain,strategy,approach,genome,score,info,generation) VALUES (?,?,?,?,?,?,?,?)",
                             (time.strftime("%Y-%m-%dT%H:%M:%S"), self.domain.name, self.strategy,
                              approach, key(g), s, json.dumps(info), gen)); conn.commit()
            if verbose and gen % max(budget // 10, 1) == 0:
                print(f"[{self.strategy}] {len(seen)}/{budget} best={best:.4g}")
        if conn: conn.close()
        return dict(best_genome=best_g, best_score=best, curve=curve, evaluations=len(seen))

# ---------- demo / CLI ----------
def demo():
    # synthetic 6-d grid with a corner optimum — exercises the v2 exhaustion escape
    grid = {f"g{i}": list(range(5)) for i in range(6)}
    fit = lambda g: -sum((v - 4) ** 2 for v in g.values()) + random.Random(str(g)).random() * 0.1
    for strat in ("greedy", "restart", "population", "random"):
        r = GaaR(GridDomain(grid, fit, "synthetic"), strategy=strat,
                 db_path="_gaar_demo.db", seed=1).run(budget=80, verbose=False)
        print(f"{strat:11s} best={r['best_score']:.3f} evals={r['evaluations']} genome={r['best_genome']}")
    print("DEMO OK (corner optimum; v1 greedy would deadlock here without the escape)")

def myuncle_search():
    import numpy as np
    from MyUncle_core import PARAMETER_GRID
    z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "myuncle_grid.npz"))
    dom = GridDomain(PARAMETER_GRID, z["eta"], "myuncle_eta")
    r = GaaR(dom, strategy="greedy", db_path="gaar_myuncle.db").run(budget=120)
    print("champion:", r["best_genome"], "eta =", round(r["best_score"], 2),
          "| ground truth eta* =", round(float(z["eta"].max()), 2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GaaR_core v2")
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--myuncle", action="store_true")
    a = ap.parse_args()
    if a.demo: demo()
    elif a.myuncle: myuncle_search()
    else: ap.print_help()
