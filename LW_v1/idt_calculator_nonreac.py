"""
idt_calculator_nonreac.py — 单阶段着火延迟时间(IDT)计算器
===========================================================
本模块用于 CH3NO2（硝基甲烷）的单阶段弱着火建模。
CH3NO2 只有一个阶段，tau_h ≈ tau_total，因此公式简化为：

    IDT = A * p^n * exp(E/T)                 (1) 单阶段阿伦尼乌斯公式

    ∫[0→t_idt] (1/IDT) dt = 1                (2) LW 积分判据

比照学长 version_2026_v3 的 idt_calculator_nonreac.py，本版本：
  - 保留 calculate_idt_hi() 作为主计算函数（即单阶段 IDT）
  - 保留 integrate_idt() 数值积分函数（不变）
  - 删除与两阶段相关的 calculate_idt_1st / calculate_idt_cf / calculate_idt 等
  - 保留 calculate_Dp / calculate_pcf / calculate_Tcf / calculate_DT / calculate_Ti
    （用于 Step 3 压升 ΔP 优化，这些与单/双阶段无关，属于热化学关系）
"""

import numpy as np
import logging


class IDTCalculator:
    """
    单阶段着火延迟时间计算器。

    参数说明：
    ----------
    A, n, E : float
        单阶段阿伦尼乌斯参数  IDT = A * p^n * exp(E/T)
    Teq, k, w : float
        压升模型参数  ΔT_cf = w * (T - Teq * p^k)
    C0 : float
        平滑系数，防止 D_Tcf 取负时开根号出错
    xf : float
        反推温度权重因子  Ti = xf * (Tcf - T) + T
    """

    def __init__(self, A, n, E, Teq, k, w, C0, xf):
        self.A = A
        self.n = n
        self.E = E
        self.Teq = Teq
        self.k = k
        self.w = w
        self.C0 = C0
        self.xf = xf

    # ============================
    #  核心：单阶段 IDT 计算
    # ============================

    def calculate_idt(self, p, T):
        """
        单阶段阿伦尼乌斯公式：IDT = A * p^n * exp(E/T)

        参数:
            p : array-like  压力 (bar)
            T : array-like  温度 (K)
        返回:
            array-like  瞬时 IDT (ms)
        """
        return self.A * p ** self.n * np.exp(self.E / T)

    # 保留别名便于与旧代码兼容（本版本中 calculate_idt_hi = calculate_idt）
    def calculate_idt_hi(self, p, T):
        """别名，等于 calculate_idt()"""
        return self.calculate_idt(p, T)

    # ============================
    #  压升 ΔP 相关（热化学关系，与阶段数无关）
    # ============================

    def calculate_DT(self, p, T):
        """
        计算第一次点火后的温差驱动力 ΔT_cf。
        ΔT_cf = w * (T - Teq * p^k)
        """
        return self.w * (T - self.Teq * p ** self.k)

    def calculate_Tcf(self, T, D_Tcf):
        """
        由温差驱动力 ΔT_cf 计算点火后火焰锋温度 Tcf。
        Tcf = T + 0.5 * (D_Tcf + sqrt(D_Tcf^2 + C0))
        C0 保证 D_Tcf 为负时取 sqrt 不出错（平滑过渡）。
        """
        return T + 0.5 * (D_Tcf + (D_Tcf ** 2 + self.C0) ** 0.5)

    def calculate_pcf(self, p, T, D_Tcf):
        """由 Tcf 反推点火后压力 pcf = Tcf/T * p"""
        Tcf = self.calculate_Tcf(T, D_Tcf)
        return Tcf / T * p

    def calculate_Dp(self, p, T):
        """
        计算一次点火后的压力升高 ΔP。
        ΔP = pcf - p，其中 pcf = Tcf/T * p
        """
        D_Tcf = self.calculate_DT(p, T)
        Tcf = self.calculate_Tcf(T, D_Tcf)
        pcf = Tcf / T * p
        return pcf - p

    def calculate_Ti(self, Tcf, T):
        """
        反推温度 Ti，用于后续燃烧阶段。
        Ti = xf * (Tcf - T) + T
        """
        return self.xf * (Tcf - T) + T


# ============================
#  数值积分工具函数（全局）
# ============================

def integrate_idt(time, idt):
    """
    对 1/IDT 做数值积分（梯形法则累积积分）。

    积分: I(t) = ∫[0→t] 1/IDT(t') dt'

    当 I(t_idt) ≈ 1 的时刻即为着火时刻（LW 判据）。

    参数:
        time : array-like  时间数组 (ms)
        idt  : array-like  各时刻的瞬时 IDT (ms)
    返回:
        integral : array-like  累积积分数组（与 time 等长）
    """
    time_step = np.diff(time)
    # 防止除零：将接近 0 的 IDT 替换为极小正数
    idt_nonzero = np.where(np.abs(idt[:-1]) < 1e-10, 1e-10, idt[:-1])
    try:
        integral = np.cumsum(1.0 / idt_nonzero * time_step)
        # 检查 NaN / Inf
        if np.any(np.isnan(integral)) or np.any(np.isinf(integral)):
            integral = np.where(np.isnan(integral) | np.isinf(integral), 1e10, integral)
            logging.basicConfig(filename='integration_warnings.log', level=logging.WARNING,
                                format='%(asctime)s - %(levelname)s - %(message)s')
            logging.warning("NaN or Inf values in integral, replaced with 1e10")
        integral = np.concatenate([[0.0], integral])
        return integral
    except Exception:
        logging.warning("Integration failed, returned default array")
        return np.zeros_like(time)
