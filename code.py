import numpy as np
import shutil as sh
import os
import glob
# function y(x) = e^(sin(x))*x^2
def Func(x):
    ans = np.exp(np.sin(x)) * x ** 2
    return ans
# outputs number of iterations to file
def OutputNumberOfIterations(method_name, k):
    file = open("output/" + method_name + ".iter_output_tech.txt", "a")
    #file.write(method_name + " iterations number = " + str(k) + "\n")
    file.write(str(k) + "\n")
    file.close()
# outputs number of function counting
def OutputNumberOfFunc(method_name, fn):
    file = open("output/" + method_name + ".func_output_tech.txt", "a")
    #file.write(method_name + " func counting number = " + str(fn) + "\n")
    file.write(str(fn) + "\n")
    file.close()
# outputs each iteration [a, b] length
def OutputL(method_name, l):
    file = open("output/" + method_name + ".len_output_tech.txt", "a")
    #file.write(method_name + " current [a, b] length = " + str(l) + "\n")
    file.write(str(l) + "\n")
    file.close()
# dichotomy's method
def Dichotomy(a, b, epsilon):
    k = 0 # iteration number
    fn = 0 # function counting number
    ans = [(a, b)]
    delta = epsilon / 3
    OutputL("Dychotomy", 1000)
    OutputL("Dychotomy", abs(b - a))
    while (abs(b - a) >= epsilon):
        k = k + 1
        x1 = (a + b) / 2 - delta
        x2 = (a + b) / 2 + delta
        f1 = Func(x1)
        f2 = Func(x2)
        fn = fn + 2
        if f1 == f2:
            a = x1
            b = x2
        elif f1 < f2:
            b = x2
        else:
            a = x1
        ans.append((a, b))
        OutputL("Dychotomy", abs(b - a))
    OutputNumberOfIterations("Dychotomy", k)
    OutputNumberOfFunc("Dychotomy", fn)
    return ans
# golden ratio method
def GoldenRatio(a, b, epsilon):
    k = 1 # iteration number
    fn = 0 # func counting number
    ans = [(a, b)]
    tau = (np.sqrt(5) - 1) / 2
    x1 = a + (b - a) * (1 - tau)
    x2 = a + (b - a) * tau
    f1 = Func(x1)
    f2 = Func(x2)
    fn = fn + 2
    OutputL("GoldenRatio", 1000)
    OutputL("GoldenRatio", abs(b - a))
    while (abs(b - a) >= epsilon):
        k = k + 1
        if f1 <= f2:
            b = x2
            x2 = x1
            x1 = a + (b - a) * (1 - tau)
            f2 = f1
            f1 = Func(x1)
            fn = fn + 1
        else:
            a = x1
            x1 = x2
            x2 = a + (b - a) * tau
            f1 = f2
            f2 = Func(x2)
            fn = fn + 1
        ans.append((a, b))
        OutputL("GoldenRatio", abs(b - a))
    OutputNumberOfIterations("GoldenRatio", k)
    OutputNumberOfFunc("GoldenRatio", fn)
    return ans
# Get Fibonach number
def FibonachiNumber(n):
    n = n + 1
    f = 1 / np.sqrt(5) * (((1 + np.sqrt(5)) / 2) ** n - ((1 - np.sqrt(5)) /2) ** n)
    return f
# Fibonachi method
def Fibonachi(a, b, epsilon):
    k = 1 # iteration number
    fn = 0 # func counting number
    ans = [(a, b)]
    n = 1
    while (FibonachiNumber(n) <= (b - a) / epsilon):
        n = n + 1
    x1 = a + (b - a) * FibonachiNumber(n - 2) / FibonachiNumber(n)
    x2 = a + (b - a) * FibonachiNumber(n - 1) / FibonachiNumber(n)
    f1 = Func(x1)
    f2 = Func(x2)
    fn = fn + 2
    OutputL("Fibonachi", 1000)
    OutputL("Fibonachi", abs(b - a))
    while(n > 2):
        k = k + 1
        if (f1 < f2):
            b = x2
            x2 = x1
            f2 = f1
            n=n-1
            x1 = a + (b - a) * FibonachiNumber(n - 2) / FibonachiNumber(n)
            f1 = Func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            n=n-1
            x2 = a + (b - a) * FibonachiNumber(n - 1) / FibonachiNumber(n)
            f2 = Func(x2)
        fn = fn + 1
        OutputL("Fibonachi", abs(b - a))
        ans.append((a, b))
    OutputNumberOfIterations("Fibonachi", k)
    OutputNumberOfFunc("Fibonachi", fn)
    return ans
# Parabol method
def Parabol(a, b, epsilon):
    k = 1 # iteration number
    fn = 0 # func counting number
    ans = [(a, b)]
    x1 = a
    f1 = Func(x1)
    x3 = b
    f3 = Func(x3)
    fn = fn + 2
    OutputL("Parabol", 1000)
    OutputL("Parabol", abs(x1 - x3))
    while (abs(x3 - x1) >= epsilon):
        k = k + 1
        x2 = (x1 + x3) / 2
        f2 = Func(x2)
        u = x2 - (((x2 - x1) ** 2)*(f2 - f3) - ((x2 - x3) ** 2)*(f2 - f1))/(2*((x2 - x1)*(f2 - f3) - (x2 - x3)*(f2 - f1)))
        fu = Func(u)
        fn = fn + 2
        if (x1 < u and u < x2):
            x3 = x2
            x2 = u
            f3 = f2
            f2 = fu
        else:
            x1 = x2
            x2 = u
            f1 = f2
            f2 = fu
        OutputL("Parabol", abs(x1 - x3))
        ans.append((x1, x3))
    OutputNumberOfIterations("Parabol", k)
    OutputNumberOfFunc("Parabol", fn)
    return ans
# Brent method
def Brent(a, c, epsilon):
    k_iter = 1 # iteration number
    f_iter = 0 # func counting number
    epsilon = epsilon / 2
    x = w = v = (a + c) / 2
    K = (3 - np.sqrt(5))/2
    fx = fw = fv = Func(x)
    f_iter = f_iter + 1
    d = e = c - a
    u = 0
    ans = [(a, c)]
    OutputL("Brent", 1000)
    OutputL("Brent", abs(a - c))
    while (abs(c - a) / 2 > epsilon):
        k_iter = k_iter + 1
        g = e
        e = d
        flag = 0
        # if points x, v, w and func fx, fw, fv are different
        if (x != w and x != v and w != v and fx != fw and fx != fv and fw != fv):
            # parabol approximation 
            x1 = min(x, min(w, v))
            x2 = x3 = f1 = f2 = f3 = 0
            if (x1 == x):
                f1 = fx
            elif (x1 == v):
                f1 = fv
            else :
                f1 = fw
            x3 = max(x, max(w, v))
            if (x3 == x):
                f3 = fx
            elif (x3 == v):
                f3 = fv
            else :
                f3 = fw
            if (x != x1 and x != x3): 
                x2 = x
                f2 = fx
            elif (w != x1 and w != x3):
                x2 = w
                f2 = fw
            elif (v != x1 and v != x3):
                x2 = v
                f2 = fv
            if (x1 < x2 and x2 < x3 and f1 > f2 and f2 < f3):
                # take u
                flag = 1
                u = x2 - (((x2 - x1) ** 2)*(f2 - f3)-((x2 - x3) ** 2)*(f2 - f1))/(2*((x2 - x1)*(f2 - f3)-(x2 - x3)*(f2 - f1)))
        if (flag == 1 and a + epsilon < u and u < c - epsilon and abs(u - x) < g / 2):
            e = abs(u - x)
        else: # if u wasn't taken priviously
            if (x < (c + a) / 2):
                # GoldenRation [x, c]
                u = x + K * (c - x)
                e = c - x
            else:
                # GoldenRation [a, x]
                u = x - K * (x - a)
                e = x - a 
        if abs(u - x) < epsilon:
            # set min u between u and x
            u = x + np.sign(u - x) * epsilon
        # count func(u)
        fu = Func(u)
        f_iter = f_iter + 1
        if (fu <= fx):
            if (u >= x):
                a = x
            else: 
                c = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        else:
            if (u >= x):
                c = u
            else:
                a = u
            if (fu <= fw or w == x):
                v = w
                w = u
                fv = fw
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu
        OutputL("Brent", abs(a - c))
        ans.append((a, c))
    OutputNumberOfIterations("Brent", k_iter)
    OutputNumberOfFunc("Brent", f_iter)         
    return ans
#
#
#
# Testing
files = glob.glob("output/*")
for f in files:
    f1 = open(f, "w")
    f1.write("")
    f1.close()
# file = open("output/Dychotomy.output_tech.txt", "w")
# file.write("")
# file.close()
# file = open("output/GoldenRatio.output_tech.txt", "w")
# file.write("")
# file.close()
# file = open("output/Fibonachi.output_tech.txt", "w")
# file.write("")
# file.close()
# file = open("output/Parabol.output_tech.txt", "w")
# file.write("")
# file.close()
# file = open("output/Brent.output_tech.txt", "w")
# file.write("")
# file.close()
test_data = [[-1, 1], [-2, 1], [-5, 2], [-5, 0], [-2, 0], [0, 1], [0, 2], [2.5, 7.5]]
for i in range(len(test_data)):
    file = open("output/output_tech.txt", "a")
    file.write("Test" + str(i + 1) + " ")
    file.close()
    a = test_data[i][0]
    b = test_data[i][1]
    epsilon = 1
    for j in range(9):
        epsilon = epsilon / 10
        file = open("output/output_tech.txt", "a")
        file.write("with epsilon = " + str(epsilon) + "\n")
        file.close()
        ans1_dych = Dichotomy(a, b, epsilon)
        ans1_gr = GoldenRatio(a, b, epsilon)
        ans1_fib = Fibonachi(a, b, epsilon)
        ans1_par = Parabol(a, b, epsilon)
        ans1_brent = Brent(a, b, epsilon)

 
