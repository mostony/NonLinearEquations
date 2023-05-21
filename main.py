import random
import scipy.optimize as opt
from solver import Solver, IntervalSolver, distance
import numpy as np
import time
from interval import interval, inf, imath
from solver import Solver, IntervalSolver
import math

np.random.seed(414)


def test_simple():
    def f(x):
        # x − y + 1 = 0
        # x^2 - y + 1 = 0
        # sols (1, 2) and (0, 1)
        return [x[0] - x[1] + 1, x[0] ** 2 - x[1] + 1]

    def f_prime(x):
        return [[1, -1],
                [2 * x[0], -1]]

    root = opt.fsolve(f, (1, 1), args=(), fprime=f_prime)
    print("root of function from fsolve : ", root[0], root[1], '; Value of function in root :', f(root))

    root = Solver(f, (1, 1), f_prime, debug=False).newton()

    print("root of function from mine solver : ", root[0], root[1], '; Value of function in root :', f(root))


def test_another_simple():
    def f(x):
        # 3x1 - cos(x2x3)-1/2=0
        # x1^2 - 81(x2+0.1)^2+sin(x3)+1.06=0
        # e^{-x1x2}+20x3+(10pi-3)/3=0
        # sols : (0.5, 0, -0.52)
        return [3 * x[0] - math.cos(x[1] * x[2]) - 1 / 2,
                x[0]**2-81*(x[1]+0.1)**2 + math.sin(x[2]) + 1.06,
                math.exp(-x[0] * x[1]) + 20 * x[2] + (10 * math.pi - 3) / 3]

    def f_prime(x):
        return [[3, x[2] * math.sin(x[1] * x[2]), x[1] * math.sin(x[1] * x[2])],
                [2 * x[0], -162 * (x[1] + 0.1), math.cos(x[2])],
                 [-x[1] * math.exp(-x[0] * x[1]), -x[0] * math.exp(-x[0] * x[1]), 20]]

    root =  opt.fsolve(f, (0.1, 0.1, -0.1), args=(), fprime=f_prime)

    print("root of function from fsolve : ", root[0], root[1], root[2], '; Value of function in root :', f(root))

    root = Solver(f, (0.1, 0.1, -0.1), f_prime).newton()

    print("root of function from mine solver (newton) : ", root[0], root[1], root[2], '; Value of function in root :', f(root))

    root = Solver(f, (0.1, 0.1, -0.1), f_prime).broyden()

    print("root of function from mine solver (broyden) : ", root[0], root[1], root[2], '; Value of function in root :', f(root))


#
# def f2(x):
#     # y = x+5
#     # y = -x^2 + 5
#     # sols (-1, 4), (0, 5)
#     return [x[0] + 5 - x[1], -x[0] ** 2 + 5 - x[1]]
#
# def f3(x):
#     # x^2 + y^2 = 5
#     # y = 3x−5
#     # sols (2, 1), (1, -2)
#     return [x[0] ^ 2 + x[1] ^ 2 - 5, x[1] - 3 * x[0] + 5]
#
#
# def F5(x):
#     return [2 * x[0] - x[1] - math.exp(-x[0]),
#             -x[0] + 2 * x[1] - math.exp(-x[1])]
#
#
# def F5_pr(x):
#     return [[2 + math.exp(-x[0]), -1], [-1, 2 + math.exp(-x[1])]]
#
#
# Solver5 = Solver(F5, (5, 5), F5_pr)
# x = Solver5.Run()
#
# print("F5 mine : ", x, F5(x))
#
# print('-------')


def test_something():
    k = interval([0, 1], [2, 3], [10, 15])
    x = interval([0, 1], [2, 3], [10, 15])

    a = interval([0, 1])
    b = interval([0, 1])
    print(a + b)

    print(k + x)
    print(interval([1, 2]) + interval([4, 5]))


def test_interval_small():
    def f(x):
        return [x[0] ** 2 + x[1] ** 2 - 1, x[0] - x[1] ** 2]

    def f_prime(x):
        return [[2 * x[0], 2 * x[1]], [1, -2 * x[1]]]

    x = Solver(f, (1, 1), f_prime).newton()

    print("f mine : ", x, f(x))

    print('─' * 100)

    x = IntervalSolver(f, (interval([0.5, 0.8]), interval([0.6, 0.9])), f_prime).krawczyk()
    ans = [x[i][0].inf for i in range(len(x))]
    print("f mine : ", ans, f(ans))

    print('─' * 100)

    x = IntervalSolver(f, (interval([0.5, 1.0]), interval([0.5, 1])), f_prime).krawczyk()
    ans = [x[i][0].inf for i in range(len(x))]
    print("f mine : ", ans, f(ans))

    print('─' * 100)


def test_big():
    def f(x):
        n = len(x)
        assert n > 3
        ret = [0] * n
        for i in range(n):
            ret[i] = 3 * x[i] - 2 * x[i] * x[i] - + 1
            if i > 0:
                ret -= x[i - 1]
            if i + 1 < n:
                ret -= 2 * x[i + 1]
        return ret

    def f_prime(x):
        n = len(x)
        assert n > 3
        ret = [[0] * n for i in range(n)]
        ret[0][0] = 3 - 4 * x[0]
        ret[0][1] = -2

        ret[n - 1][n - 1] = 3 - 4 * x[n - 1]
        ret[n - 1][n - 2] = -1
        for i in range(1, n - 1):
            ret[i][i - 1] = -1
            ret[i][i] = 3 - 4 * x[i]
            ret[i][i + 1] = -2
        return ret

    N = 4
    x_start = [-1] * N
    # print(f_prime(x_start))
    # root = opt.newton(f, x0=x_start, fprime=f_prime)
    # print(root, f(root))

    x = Solver(f, x_start, f_prime).new()

    print("F5 mine : ", x, f(x))


def test_big_exp():
    def f(x):
        n = len(x)
        # assert n > 3
        ret = [0] * n
        ret[0] = math.exp(x[0]) - 1
        for i in range(1, n):
            ret[i] = (i + 1) / 10 * (math.exp(x[i]) + x[i - 1] - 1)
        return ret

    def f_interval(x):
        n = len(x)
        # assert n > 3
        ret = [interval(0, 0)] * n

        ret[0] = imath.exp(x[0]) - 1
        for i in range(1, n):
            ret[i] = (i + 1) / 10 * (imath.exp(x[i]) + x[i - 1] - 1)
        return ret

    def f_prime(x):
        n = len(x)
        # assert n > 3
        ret = [[0] * n for i in range(n)]
        ret[0][0] = math.exp(x[0])
        for i in range(1, n):
            ret[i][i - 1] = i / 10
            ret[i][i] = i / 10 * math.exp(x[i])
        return ret

    def f_prime_interval(x):
        n = len(x)
        # assert n > 3
        ret = [[0] * n for i in range(n)]
        ret[0][0] = imath.exp(x[0])
        for i in range(1, n):
            ret[i][i - 1] = i / 10
            ret[i][i] = i / 10 * imath.exp(x[i])
        return ret

    N = 10

    def usual_solve():
        x_start = np.random.normal(size=N)
        x_start = np.random.uniform(low=0, high=0.01, size=N)
        x_start = [0.01] * N
        x = Solver(f, x_start, f_prime, max_iter=100, debug=False).newton()
        print(f'usual diff {distance(np.zeros(N), f(x))}')
        print(f'root {x}')

        # mn = min(x)
        # mx = max(x)
        # T = max(mx, -mn)
        # d = mx - mn
        #
        # print("T is ", T)
        # # x_start = [interval([-T + T / 2, T - T / 2]) for i in range(N)]
        # x_start = [interval([ ]) for i in range(N)]
        #
        # x = IntervalSolver(f_interval, x_start, f_prime_interval, max_iter=100, debug=False).krawczyk()
        # print(distance(np.zeros(N), f_interval(x)))

    def interval_solve():
        x_start = [interval([-0.01, 0.01]) for i in range(N)]
        x = IntervalSolver(f_interval, x_start, f_prime_interval, max_iter=100, debug=True).krawczyk()
        # print(x, f_interval(x))
        print(f'interval diff {distance(np.zeros(N), f_interval(x))}')

    start = time.time()

    usual_solve()
    interval_solve()

    end = time.time()
    print(f'Elapsed time : {end - start}')

    # print(f_prime(x_start))
    # root = opt.newton(f, x0=x_start, fprime=f_prime)
    # print(root, f(root))
    #
    # x = Solver(f, x_start, f_prime, max_iter=1000).Run()

    # print("{root, value} : ", x, f(x))


def test_rosenbrock():
    # Extended - Rosenbrock
    # function
    def f(x):
        n = len(x)
        assert n > 3
        ret = [0] * n

        for i in range(int(n / 2)):
            ret[2 * i] = 10 * (x[2 * i + 1] - x[2 * i] ** 2)
            ret[2 * i + 1] = 1 - x[2 * i]
        return ret

    def f_prime(x):
        n = len(x)
        assert n > 3
        ret = [[0] * n for i in range(n)]
        k = int(n / 2)
        for i in range(0, k):
            ret[2 * i][2 * i + 1] = 10
            ret[2 * i][2 * i] = -20 * x[2 * i]

            ret[2 * i + 1][2 * i] = -1
        return ret

    N = 10

    assert N % 2 == 0

    def usual_solve():
        x_start = np.random.uniform(low=0, high=N, size=N)
        x = Solver(f, x_start, f_prime, max_iter=10, debug=True).broyden()
        print(distance(np.zeros(N), f(x)))

    def interval_solve():
        x_start = [interval([0.95, 1.05]) for i in range(N)]
        x = IntervalSolver(f, x_start, f_prime, max_iter=1000, debug=True).krawczyk()
        # print(x, f_interval(x))
        print(distance(np.zeros(N), f(x)))

    start = time.time()

    usual_solve()
    interval_solve()

    end = time.time()
    print(f'Elapsed time : {end - start}')

# test_another_simple()
# test_simple()
# test_something()
# test_interval_small()
# test_big()
# test_big_exp()
# test_rosenbrock()
