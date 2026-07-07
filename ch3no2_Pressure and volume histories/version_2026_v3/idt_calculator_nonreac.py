import numpy as np
import logging

class IDTCalculator:
    def __init__(self, A1, n1, E1, Ah, nh, Eh, Teq, k, w, C0, xf):
        self.A1 = A1
        self.n1 = n1
        self.E1 = E1
        self.Ah = Ah
        self.nh = nh
        self.Eh = Eh
        self.Teq = Teq
        self.k = k
        self.w = w
        self.C0 = C0
        self.xf = xf

    def calculate_idt_1st(self, p, T):
        """Calculate first-stage IDT"""
        return self.A1 * p ** self.n1 * np.exp(self.E1 / T)

    def calculate_idt_hi(self, p, T):
        """Calculate high-temperature IDT with overflow protection"""
        return self.Ah * p ** self.nh * np.exp(self.Eh / T)
        

    def calculate_Dp(self, p, T):
        """Calculate The pressure rise at 1st stage ignition"""
        D_Tcf = self.w * (T - self.Teq * p ** self.k)
        Tcf = T + 0.5 * (D_Tcf + (D_Tcf ** 2 + self.C0) ** 0.5)
        # Tcf = T + D_Tcf
        pcf = Tcf / T * p
        return pcf - p

    def calculate_pcf(self, p, T, D_Tcf):
        """Calculate pcf"""
        Tcf = T + 0.5 * (D_Tcf + (D_Tcf ** 2 + self.C0) ** 0.5)
        pcf = Tcf / T * p
        return pcf

    def calculate_Tcf(self, T, D_Tcf):
        """Calculate Tcf"""
        Tcf = T + 0.5 * (D_Tcf + (D_Tcf ** 2 + self.C0) ** 0.5)
        return Tcf

    def calculate_Ti(self, Tcf, T):
        """Calculate Ti"""
        Ti = self.xf * (Tcf - T) + T
        return Ti

    def calculate_DT(self, p, T):
        """Calculate The temperature rise at 1st stage ignition"""
        D_Tcf = self.w * (T - self.Teq * p ** self.k)
        return D_Tcf



    def calculate_idt_cf(self, p, T):
        """Calculate cross-term IDT"""
        D_Tcf = self.w * (T - self.Teq * p ** self.k)
        Tcf = T + 0.5 * (D_Tcf + (D_Tcf ** 2 + self.C0) ** 0.5)
        pcf = Tcf / T * p
        Ti = self.xf * (Tcf - T) + T
        #Tcf = self.calculate_Tcf(self, p, T)
        #pcf = self.calculate_pcf(self, p, T)
        return self.calculate_idt_hi(pcf, Ti)


    def calculate_idt(self, p, T):
        """Calculate total IDT"""
        t1 = self.calculate_idt_1st(p, T)
        t_hi = self.calculate_idt_hi(p, T)
        t_cf = self.calculate_idt_cf(p, T)
        return t1 + t_cf * (1 - t1 / t_hi)


def integrate_idt(time, idt):
    """Integrate 1/IDT over time with zero division protection"""
    time_step = np.diff(time)
    # To prevent division by zero errors, replace idt values nearly equal to zero with a small positive number.
    idt_nonzero = np.where(np.abs(idt[:-1]) < 1e-10, 1e-10, idt[:-1])
    try:
        integral = np.cumsum(1 / idt_nonzero * time_step)
        # Check for NaN or infinite values.
        if np.any(np.isnan(integral)) or np.any(np.isinf(integral)):
            # Replace non-finite values with large positive numbers
            integral = np.where(np.isnan(integral) | np.isinf(integral), 1e10, integral)
            print("Warning: NaN or Inf values in integral, replaced with 1e10")
            # Configure Logging
            logging.basicConfig(filename='integration_warnings.log', level=logging.WARNING,
                               format='%(asctime)s - %(levelname)s - %(message)s')
            logging.warning("NaN or Inf values in integral, replaced with 1e10")
        integral = np.concatenate([[0], integral])
        return integral
    except:
        # Return the default array in the event of any exception.
        return np.zeros_like(time)
        print("Warning: Integration failed, returned default array")
        logging.warning("Integration failed, returned default array")  

