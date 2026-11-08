"""DQN (PyTorch): jaringan MLP Q(s; theta) dengan satu keluaran per aksi diskret.

Komponen:
  - replay buffer utama (semua transisi) + buffer SUKSES (transisi dari episode yang mencapai
    goal), supaya pengalaman goal yang jarang tidak tenggelam oleh banyak transisi tabrakan
  - target network dengan soft update (theta_target <- tau*theta + (1-tau)*theta_target)
  - Double DQN: aksi dipilih jaringan online, nilainya dievaluasi target network
        y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))     (tanpa suku kedua di state akhir)
  - target dipotong ke [Y_MIN, Y_MAX] supaya Q tidak divergen
  - loss Huber, gradient clipping, learning rate 3e-4
"""
import numpy as np
import torch
import torch.nn as nn

Y_MIN, Y_MAX = -15.0, 8.0     # batas target (skala imbalan x0,05): goal +100 -> +5 (+bonus sel baru <~2), tabrakan -5, hazard terakumulasi <~ -12
TIE_TOL = 1e-3      # selisih Q (skala imbalan x0,05) yang dianggap seri saat memilih aksi greedy


class QNet(nn.Module):
    def __init__(self, n_in, n_act, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_act))
        # Lapisan keluaran dikecilkan: Q awal ~0 untuk semua aksi, jadi jaringan baru tidak
        # berpihak ke satu aksi (sebelumnya argmax jatuh ke satu aksi hampir di semua state).
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Buffer melingkar: transisi terlama ditimpa saat penuh."""

    def __init__(self, cap, n_in):
        self.cap, self.size, self.pos = cap, 0, 0
        self.s = np.zeros((cap, n_in), dtype=np.float32)
        self.a = np.zeros(cap, dtype=np.int64)
        self.r = np.zeros(cap, dtype=np.float32)
        self.s2 = np.zeros((cap, n_in), dtype=np.float32)
        self.d = np.zeros(cap, dtype=np.float32)

    def add(self, s, a, r, s2, d):
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.pos = (self.pos + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, k, rng):
        idx = np.fromiter((rng.randrange(self.size) for _ in range(k)), dtype=np.int64)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class DQNAgent:
    def __init__(self, n_in, n_act, rng, lr=3e-4, gamma=0.99, buffer_size=100_000,
                 success_size=20_000, success_frac=0.25, batch=64, tau=0.005,
                 warmup=1000, reward_scale=0.05, n_policy=None, n_step=1):
        torch.set_num_threads(2)
        # n_policy: hanya aksi 0..n_policy-1 yang boleh DIPILIH kebijakan (epsilon-acak, greedy, dan max di
        # target). Aksi sisanya (indeks >= n_policy) hanya dijalankan lapisan pengaman di luar DQN.
        self.n_act, self.rng = n_act, rng
        self.n_policy = n_act if n_policy is None else n_policy
        self.gamma, self.batch, self.tau = gamma, batch, tau
        self.n_step = n_step            # n > 1: target n-langkah (lihat end_episode); n = 1: TD biasa
        self.warmup, self.reward_scale, self.success_frac = warmup, reward_scale, success_frac
        self.q = QNet(n_in, n_act)
        self.target = QNet(n_in, n_act)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.main = ReplayBuffer(buffer_size, n_in)
        self.succ = ReplayBuffer(success_size, n_in)
        self._episode = []                  # transisi episode berjalan (disalin ke succ bila goal)
        self.updates = 0
        self.last_loss = 0.0
        self.last_q = 0.0
        self.last_greedy = False

    @property
    def size(self):
        return self.main.size

    def act(self, feat, eps):
        """epsilon-greedy atas indeks aksi. Bagian greedy = argmax keluaran jaringan; aksi yang
        Q-nya berselisih <= TIE_TOL dari yang terbesar dianggap seri dan dipilih acak
        (seperti Q-learning). self.last_greedy mencatat apakah keputusan ini greedy."""
        if self.rng.random() < eps:
            self.last_greedy = False
            return self.rng.randrange(self.n_policy)
        self.last_greedy = True
        with torch.no_grad():
            qv = self.q(torch.from_numpy(feat).unsqueeze(0))[0].numpy()[:self.n_policy]
        best = np.flatnonzero(qv >= qv.max() - TIE_TOL)
        return int(self.rng.choice(list(best)))

    def remember(self, s, a, r, s2, done):
        t = (s, a, r * self.reward_scale, s2, float(done))
        self._episode.append(t)
        if self.n_step == 1:
            self.main.add(*t)

    def end_episode(self, success):
        """Panggil di akhir tiap episode. n_step = 1: bila sukses, transisi episode masuk buffer sukses.
        n_step > 1: seluruh episode diubah menjadi transisi n-langkah lalu dimasukkan ke buffer utama (dan buffer
        sukses bila sukses): r = sum_{k<n} gamma^k r_{t+k}, s' = state setelah n langkah, done = episode berakhir
        dalam n langkah (tanpa bootstrap). Pembaruan memakai gamma**n. Target 1-langkah dengan gamma 0,995
        memperbesar bias maks-Q ~200x sehingga Q jenuh di batas atas; target n-langkah memutus itu."""
        ep = self._episode
        if self.n_step == 1:
            if success:
                for t in ep:
                    self.succ.add(*t)
        else:
            L, n = len(ep), self.n_step
            for i in range(L):
                m = min(n, L - i)
                g = 0.0
                for k in range(m):
                    g += (self.gamma ** k) * ep[i + k][2]
                last = ep[min(i + n, L) - 1]
                t = (ep[i][0], ep[i][1], g, last[3], 1.0 if i + n >= L else 0.0)
                self.main.add(*t)
                if success:
                    self.succ.add(*t)
        self._episode = []

    def learn(self):
        if self.main.size < max(self.warmup, self.batch):
            return
        n_succ = int(self.batch * self.success_frac)
        if self.succ.size < max(n_succ, self.batch):      # tunggu sampai buffer sukses cukup
            n_succ = 0
        parts = [self.main.sample(self.batch - n_succ, self.rng)]
        if n_succ:
            parts.append(self.succ.sample(n_succ, self.rng))
        s, a, r, s2, d = (torch.from_numpy(np.concatenate(x)) for x in zip(*parts))
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_star = self.q(s2)[:, :self.n_policy].argmax(1, keepdim=True)    # Double DQN (hanya aksi kebijakan)
            y = r + (self.gamma ** self.n_step) * (1.0 - d) * self.target(s2).gather(1, a_star).squeeze(1)
            y = y.clamp(Y_MIN, Y_MAX)       # cegah Q membengkak (bootstrapping tanpa batas)
        loss = nn.functional.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.last_loss = float(loss.item())
        self.last_q = float(q_sa.detach().mean())
        self.updates += 1
        with torch.no_grad():                                                 # soft update
            for pt, p in zip(self.target.parameters(), self.q.parameters()):
                pt.mul_(1.0 - self.tau).add_(p, alpha=self.tau)

    def state_dict(self):
        return {k: v.clone() for k, v in self.q.state_dict().items()}

    def load_state_dict(self, sd):
        self.q.load_state_dict(sd)
        self.target.load_state_dict(sd)
