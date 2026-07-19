# -*- coding: utf-8 -*-
"""
행렬과 인공지능 — 시뮬레이터
2026 수학-정보 융합 프로젝트(기초) · 당곡고등학교
실행:  streamlit run app.py
"""
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="행렬과 인공지능", page_icon="🔢", layout="wide")

# ---------------------------------------------------------------- 공통
def det2(A):
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

def inv2(A):
    d = det2(A)
    if abs(d) < 1e-10:
        return None
    return np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / d

def show_matrix(A, name="A"):
    r = lambda v: f"{v:.3g}"
    st.latex(
        rf"{name}=\begin{{pmatrix}} {r(A[0,0])} & {r(A[0,1])} \\ "
        rf"{r(A[1,0])} & {r(A[1,1])} \end{{pmatrix}}"
    )

# 글자 F (방향·뒤집힘 확인용)
F = np.array([
    [0.00, 0.00], [0.30, 0.00], [0.30, 0.40], [0.68, 0.40], [0.68, 0.58],
    [0.30, 0.58], [0.30, 0.82], [0.90, 0.82], [0.90, 1.00], [0.00, 1.00],
])

st.title("🔢 행렬과 인공지능")
st.caption("2026 수학–정보 융합 프로젝트(기초) · 당곡고등학교")

tabs = st.tabs([
    "1. 두 점이 가는 곳",
    "2. 일차변환 실험실",
    "3. 연립일차방정식",
    "4. 행렬 암호",
])

# ================================================================ 1. 두 점
with tabs[0]:
    st.header("행렬은 두 점을 어디로 보내는가")
    st.caption("행렬 A는 점 (1,0)을 u=(a,c)로, 점 (0,1)을 v=(b,d)로 보냅니다. "
               "두 화살표가 만드는 평행사변형의 넓이가 곧 행렬식입니다.")

    # 프리셋은 슬라이더 키를 직접 바꿔야 반영됨
    for _k, _v in [("sl_a", 1.0), ("sl_c", 0.0), ("sl_b", 0.5), ("sl_d", 1.0)]:
        st.session_state.setdefault(_k, _v)

    def set_bg(a, b, c, d):
        st.session_state.sl_a = a
        st.session_state.sl_c = c
        st.session_state.sl_b = b
        st.session_state.sl_d = d

    left, right = st.columns([1, 1.5])
    with left:
        st.markdown("**프리셋**")
        p1, p2 = st.columns(2)
        p1.button("항등 E", use_container_width=True, on_click=set_bg, args=(1., 0., 0., 1.))
        p2.button("기울임체", use_container_width=True, on_click=set_bg, args=(1., .8, 0., 1.))
        p1.button("회전 53°", use_container_width=True, on_click=set_bg, args=(.6, -.8, .8, .6))
        p2.button("x축 대칭", use_container_width=True, on_click=set_bg, args=(1., 0., 0., -1.))
        p1.button("닮음 1.5배", use_container_width=True, on_click=set_bg, args=(1.5, 0., 0., 1.5))
        p2.button("역행렬 없음 (바로)", use_container_width=True, on_click=set_bg, args=(1., 2., 2., 4.))
        st.button("역행렬 없음 (과정)", use_container_width=True, on_click=set_bg,
                  args=(1., 1., 0., 1.),
                  help="누른 뒤 c 슬라이더를 0 → 1.0 으로 천천히 밀어 보세요")

        st.divider()
        a = st.slider("a  (u의 x성분)", -3.0, 3.0, step=0.1, key="sl_a")
        c = st.slider("c  (u의 y성분)", -3.0, 3.0, step=0.1, key="sl_c")
        b = st.slider("b  (v의 x성분)", -3.0, 3.0, step=0.1, key="sl_b")
        d = st.slider("d  (v의 y성분)", -3.0, 3.0, step=0.1, key="sl_d")

    M = np.array([[a, b], [c, d]], float)
    dM = det2(M)

    with right:
        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        K = 7
        for k in range(-K, K + 1):                       # 변환 전 격자
            ax.plot([k, k], [-K, K], color="#d4d4d8", lw=0.8, zorder=1)
            ax.plot([-K, K], [k, k], color="#d4d4d8", lw=0.8, zorder=1)
        for k in range(-K, K + 1):                       # 변환 후 격자
            for P in ([[k, -K], [k, K]], [[-K, k], [K, k]]):
                Q = np.array(P, float) @ M.T
                ax.plot(Q[:, 0], Q[:, 1], color="#60a5fa", lw=0.9, zorder=2)
        u, v = M[:, 0], M[:, 1]
        poly = np.array([[0, 0], u, u + v, v])           # 평행사변형
        ax.fill(poly[:, 0], poly[:, 1], color="#a78bfa", alpha=0.35, zorder=3)
        for tip in [(1, 0), (0, 1)]:                     # 원래 기준점 (점선)
            ax.annotate("", xy=tip, xytext=(0, 0), zorder=4,
                        arrowprops=dict(arrowstyle="->", lw=1.6, color="#9ca3af", ls="--"))
        ax.annotate("", xy=tuple(u), xytext=(0, 0), zorder=5,
                    arrowprops=dict(arrowstyle="->", lw=3.0, color="#dc2626"))
        ax.annotate("", xy=tuple(v), xytext=(0, 0), zorder=5,
                    arrowprops=dict(arrowstyle="->", lw=3.0, color="#16a34a"))
        ax.text(u[0] * 1.08, u[1] * 1.08, "u=(a,c)", color="#dc2626", fontsize=11, weight="bold")
        ax.text(v[0] * 1.08, v[1] * 1.08, "v=(b,d)", color="#16a34a", fontsize=11, weight="bold")
        ax.axhline(0, color="#1f2937", lw=1.0, zorder=3)
        ax.axvline(0, color="#1f2937", lw=1.0, zorder=3)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
        ax.set_xticks(range(-4, 5)); ax.set_yticks(range(-4, 5))
        ax.tick_params(labelsize=8)
        st.pyplot(fig)
        plt.close(fig)

    m1, m2, m3 = st.columns(3)
    m1.metric("det A = ad − bc", f"{dM:.3f}")
    m2.metric("평행사변형 넓이", f"{abs(dM):.3f}")
    m3.metric("방향", "유지" if dM > 1e-9 else ("뒤집힘" if dM < -1e-9 else "—"))

    # 어떤 값을 바꾸면 det = 0 이 되는지 안내
    if abs(dM) > 1e-9:
        hints = []
        if abs(b) > 1e-9 and -3.0 <= a * d / b <= 3.0:
            hints.append(f"**c 를 {a*d/b:.2f}** 로")
        if abs(a) > 1e-9 and -3.0 <= b * c / a <= 3.0:
            hints.append(f"**d 를 {b*c/a:.2f}** 로")
        if hints:
            st.caption("🎯 지금 상태에서 " + " 또는 ".join(hints)
                       + " 맞추면 det = 0 이 되어 역행렬이 존재하지 않게 됩니다.")

    if abs(dM) < 1e-9:
        st.error("**det = 0 → 역행렬이 존재하지 않습니다.**  "
                 "두 화살표가 한 직선 위에 놓여 평행사변형이 만들어지지 않고, "
                 "평면 전체가 직선 하나로 옮겨졌습니다. "
                 "서로 다른 점들이 같은 자리로 겹쳐졌으므로 원래 위치를 알 수 없습니다.")
    elif dM < 0:
        st.warning(f"**det < 0** — 방향이 뒤집혔습니다(대칭). 넓이는 {abs(dM):.3f}배입니다.")
    else:
        st.success(f"넓이가 **{dM:.3f}배**가 되었습니다. 역행렬은 이것을 "
                   f"**1/{dM:.3f} = {1/dM:.3f}배**로 되돌립니다 — "
                   f"역행렬 공식의 분모가 ad−bc 인 이유입니다.")

    with st.expander("💡 직접 확인해 보기"):
        st.markdown("""
1. **a만 키워 보세요.** 빨간 화살표만 길어집니다. → a는 점 (1,0)의 행선지
2. **b를 키워 보세요.** 초록 화살표가 오른쪽으로 눕습니다. 워드프로세서의 **기울임체**입니다.
   이때 넓이는? → 밑변도 높이도 그대로라 **변하지 않습니다** (det = 1)
3. **`역행렬 없음 (과정)` 버튼**을 누른 뒤 c를 0 → 1.0 으로 천천히 밀어 보세요.
   det가 1 → 0.8 → 0.6 → … → 0 으로 줄며 격자가 직선 하나로 눌립니다.

> 📚 점 (1,0), (0,1)처럼 기준이 되는 두 점을 대학에서는 **기저벡터**라고 부릅니다.
> 지금은 몰라도 됩니다 — 나중에 자료를 찾다 이 말이 나오면 "아, 그 두 점"이라고 생각하면 돼요.
        """)

# ================================================================ 2. 일차변환
with tabs[1]:
    st.header("일차변환 실험실 — 도형으로 확인하기")
    st.caption("책자 30쪽 『워드프로세서에서 찾는 일차변환』을 시뮬레이션합니다.")

    c0, c1, c2 = st.columns([1, 1, 1.6])
    with c0:
        preset = st.selectbox("변환 고르기", [
            "기울임체(전단)", "회전", "x축 대칭", "y축 대칭", "원점 대칭",
            "y=x 대칭", "닮음(확대/축소)", "역행렬 없음(det=0)", "직접 입력",
        ])
    with c1:
        if preset == "기울임체(전단)":
            k = st.slider("기울기 k", -1.5, 1.5, 0.4, 0.05)
            M = np.array([[1, k], [0, 1]], float)
        elif preset == "회전":
            th = st.slider("회전각 θ (도)", -180, 180, 45, 5)
            r_ = np.radians(th)
            M = np.array([[np.cos(r_), -np.sin(r_)], [np.sin(r_), np.cos(r_)]])
        elif preset == "x축 대칭":
            M = np.array([[1, 0], [0, -1]], float)
        elif preset == "y축 대칭":
            M = np.array([[-1, 0], [0, 1]], float)
        elif preset == "원점 대칭":
            M = np.array([[-1, 0], [0, -1]], float)
        elif preset == "y=x 대칭":
            M = np.array([[0, 1], [1, 0]], float)
        elif preset == "닮음(확대/축소)":
            k = st.slider("닮음비 k", -2.0, 3.0, 1.5, 0.1)
            M = np.array([[k, 0], [0, k]], float)
        elif preset == "역행렬 없음(det=0)":
            M = np.array([[1, 2], [2, 4]], float)
        else:
            cc = st.columns(2)
            a_ = cc[0].number_input("a", value=1.0, step=0.5)
            b_ = cc[1].number_input("b", value=0.4, step=0.5)
            c_ = cc[0].number_input("c", value=0.0, step=0.5)
            d_ = cc[1].number_input("d", value=1.0, step=0.5)
            M = np.array([[a_, b_], [c_, d_]], float)
        apply_inv = st.checkbox("역행렬을 한 번 더 적용 (되돌리기)")

    Q = F @ M.T
    Minv = inv2(M)
    if apply_inv and Minv is not None:
        Q = Q @ Minv.T

    with c2:
        show_matrix(M, "M")
        st.latex(rf"\det M = {det2(M):.4g}")
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        for pts, col, alp in [(F, "#94a3b8", 0.35), (Q, "#2563eb", 0.45)]:
            poly = np.vstack([pts, pts[0]])
            ax.fill(poly[:, 0], poly[:, 1], color=col, alpha=alp)
            ax.plot(poly[:, 0], poly[:, 1], color=col, lw=2)
        ax.axhline(0, color="#334155", lw=0.9)
        ax.axvline(0, color="#334155", lw=0.9)
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    if apply_inv:
        if Minv is None:
            st.error("det = 0 → **역행렬이 존재하지 않으므로 되돌릴 수 없습니다.** "
                     "도형이 선으로 눌린 것을 확인하세요.")
        else:
            st.success("M⁻¹M = E 이므로 원래 도형으로 정확히 돌아옵니다. (두 도형이 겹쳐 보입니다)")

    with st.expander("💡 관찰 과제"):
        st.markdown("""
- 대칭변환 **네 가지**(x축·y축·원점·y=x)의 det를 모아 보세요. **하나만 부호가 다릅니다.** 왜일까요?
  (힌트: 그 변환에서 F가 **거울처럼 뒤집혔나요, 그냥 돌아갔나요?**)
- 닮음 k=2 일 때 det는 왜 4일까요?
- `역행렬 없음(det=0)` 을 고르고 되돌리기를 켜 보세요. 왜 실패할까요?
        """)

# ================================================================ 3. 연립방정식
with tabs[2]:
    st.header("연립일차방정식을 역행렬로 풀기")
    st.latex(r"\begin{cases} ax+by=p \\ cx+dy=q \end{cases} \iff AX=B \iff X=A^{-1}B")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        DEFAULTS = {
            "오리와 양 (책자 24쪽)": (1, 1, 15, 2, 4, 40),
            "해가 무수히 많음": (1, 2, 3, 2, 4, 6),
            "해가 없음": (1, 2, 3, 2, 4, 7),
            "직접 입력": (3, 1, 9, 2, 1, 7),
        }
        # ⚠️ number_input에 key를 주면 위젯 상태가 value= 보다 우선한다.
        #    드롭다운이 바뀔 때 실제로 숫자칸을 갱신하려면
        #    "드롭다운이 바뀌었는지"를 직접 감지해 session_state를 덮어써야 한다.
        ex = st.selectbox("예제 불러오기", list(DEFAULTS.keys()))
        if st.session_state.get("_last_ex") != ex:
            st.session_state["_last_ex"] = ex
            a0, b0, p0, c0, d0, q0 = DEFAULTS[ex]
            st.session_state["s_a"] = float(a0)
            st.session_state["s_b"] = float(b0)
            st.session_state["s_p"] = float(p0)
            st.session_state["s_c"] = float(c0)
            st.session_state["s_d"] = float(d0)
            st.session_state["s_q"] = float(q0)

        cc = st.columns(3)
        a = cc[0].number_input("a", key="s_a")
        b = cc[1].number_input("b", key="s_b")
        p = cc[2].number_input("p", key="s_p")
        c_ = cc[0].number_input("c", key="s_c")
        d_ = cc[1].number_input("d", key="s_d")
        q = cc[2].number_input("q", key="s_q")
        if ex == "오리와 양 (책자 24쪽)":
            st.caption("머리 15개, 다리 40개. 오리 x마리, 양 y마리는?")

    A = np.array([[a, b], [c_, d_]], float)
    B = np.array([[p], [q]], float)
    dA = det2(A)

    with c2:
        st.latex(rf"\det A = {dA:.4g}")
        Ainv = inv2(A)
        if Ainv is None:
            allzero = np.allclose([a, b, c_, d_], 0)
            same = (not allzero) and np.isclose(a * q, c_ * p) and np.isclose(b * q, d_ * p)
            if allzero and not np.allclose([p, q], 0):
                same = False
            if same:
                st.warning("det = 0 이고 두 식이 같은 직선 → **해가 무수히 많습니다.** (정보 부족)")
            else:
                st.error("det = 0 이고 두 직선이 평행 → **해가 없습니다.** (정보 모순)")
        else:
            X = Ainv @ B
            st.latex(rf"X=A^{{-1}}B=\begin{{pmatrix}}{X[0,0]:.4g}\\ {X[1,0]:.4g}\end{{pmatrix}}")
            st.success(f"x = {X[0,0]:.4g},  y = {X[1,0]:.4g}")

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    xs = np.linspace(-20, 20, 400)
    for (aa, bb, pp, col) in [(a, b, p, "#2563eb"), (c_, d_, q, "#dc2626")]:
        if abs(bb) > 1e-9:
            ax.plot(xs, (pp - aa * xs) / bb, color=col, lw=2)
        elif abs(aa) > 1e-9:
            ax.axvline(pp / aa, color=col, lw=2)
    if Ainv is not None:
        X = Ainv @ B
        ax.plot(X[0, 0], X[1, 0], "o", ms=11, color="#16a34a", zorder=5)
        lim = max(6, abs(X[0, 0]) * 1.8, abs(X[1, 0]) * 1.8)
    else:
        lim = 12
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="#334155", lw=0.8); ax.axvline(0, color="#334155", lw=0.8)
    ax.grid(alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("💡 생각해 보기"):
        st.markdown("""
- det = 0 은 그래프에서 두 직선이 **평행하거나 겹친다**는 뜻입니다.
- 해가 무수히 많음 = 두 번째 식이 첫 번째 식의 상수배 → **새로운 정보가 없음**
- 해가 없음 = 두 식이 서로 모순 → **불가능한 조건**
- 즉 여기서도 **det = 0 ⟺ 역행렬이 존재하지 않는다 ⟺ 답을 하나로 정할 수 없다** 입니다.
        """)

# ================================================================ 4. 암호
with tabs[3]:
    st.header("행렬 암호 — 역행렬이 열쇠다")
    st.caption("알파벳을 A=0 ~ Z=25 숫자로 바꾸고, 2개씩 묶어 열쇠 행렬을 곱합니다(mod 26). "
               "복호화는 역행렬을 곱하면 끝 — 역행렬이 곧 열쇠입니다.")

    c1, c2 = st.columns([1, 1])
    with c1:
        key_txt = st.text_input("열쇠 행렬 (a b c d)", "3 3 2 5")
        msg = st.text_input("영문 메시지", "MATRIX").upper()
    try:
        kv = [int(v) for v in key_txt.split()]
        Km = np.array([[kv[0], kv[1]], [kv[2], kv[3]]])
    except Exception:
        Km = np.array([[3, 3], [2, 5]])

    def modinv(x, m=26):
        x %= m
        for i in range(1, m):
            if (x * i) % m == 1:
                return i
        return None

    letters = [ch for ch in msg if ch.isalpha()]
    if len(letters) % 2:
        letters.append("X")
    nums = np.array([ord(ch) - 65 for ch in letters]).reshape(-1, 2).T

    dK = int(round(det2(Km.astype(float)))) % 26
    inv_d = modinv(dK)

    with c2:
        show_matrix(Km, "K")
        st.latex(rf"\det K \equiv {dK} \pmod{{26}}")
        if inv_d is None:
            st.error(f"det K = {dK} 는 26과 서로소가 아니어서 **mod 26에서 역행렬이 없습니다.** "
                     "→ 이 열쇠로는 암호를 만들어도 **복호화할 수 없습니다.**")
        else:
            enc = (Km @ nums) % 26
            cipher = "".join(chr(v + 65) for v in enc.T.ravel())
            adj = np.array([[Km[1, 1], -Km[0, 1]], [-Km[1, 0], Km[0, 0]]])
            Kinv = (inv_d * adj) % 26
            dec = (Kinv @ enc) % 26
            plain = "".join(chr(v + 65) for v in dec.T.ravel())
            st.success(f"암호문: **{cipher}**")
            show_matrix(Kinv, "K^{-1}\\ (\\mathrm{mod}\\ 26)")
            st.info(f"복호문: **{plain}**")

    st.divider()
    st.markdown("""
**여기서도 같은 원리입니다.**
역행렬이 있으면 암호를 되돌릴 수 있고(복호화), 없으면 만든 사람조차 못 되돌립니다.

**그런데 왜 현대 암호는 행렬을 안 쓸까요?**
평문–암호문 쌍을 두 쌍만 알면 연립방정식을 풀어 열쇠가 그대로 노출되기 때문입니다.
그래서 현대 암호(RSA 등)는 *"수학적으로는 되돌릴 수 있지만 수백 년이 걸리는"* 구조를 씁니다.

> 🔎 자세한 내용은 2학년 **정보보안 프로그램**에서 다룹니다.
    """)

st.divider()
st.caption("만든 이: 김하은 | 2026 수학정보 융합캠프")
