import scipy.optimize as opt
from solver import Solver, IntervalSolver, distance
import numpy as np
import time
from interval import interval, inf, imath
from solver import Solver, IntervalSolver
import math


def test_broyden_tridiagonal():
    def f(x):
        n = len(x)
        assert n > 3
        ret = [0] * n
        for i in range(n):
            ret[i] = x[i] * (3 - 0.5 * x[i]) + 1
            if i > 0:
                ret[i] -= x[i - 1]
            if i + 1 < n:
                ret[i] -= 2 * x[i + 1]
        return ret

    def f_prime(x):
        n = len(x)
        assert n > 3
        ret = [[0] * n for i in range(n)]
        ret[0][0] = 3 - x[0]
        ret[0][1] = -2

        ret[n - 1][n - 1] = 3 - x[n - 1]
        ret[n - 1][n - 2] = -1
        for i in range(1, n - 1):
            ret[i][i - 1] = -1
            ret[i][i] = 3 - x[i]
            ret[i][i + 1] = -2
        return ret

    def f_interval(x):
        n = len(x)
        assert n > 3
        ret = [interval(0, 0)] * n
        for i in range(n):
            ret[i] = x[i] * (3 - 0.5 * x[i]) + 1
            if i > 0:
                ret[i] -= x[i - 1]
            if i + 1 < n:
                ret[i] -= 2 * x[i + 1]
        return ret

    def f_interval_prime(x):
        n = len(x)
        assert n > 3
        ret = [[interval(0, 0) for a in range(n)] for b in range(n)]
        ret[0][0] = 3 - x[0]
        ret[0][1] = -2

        ret[n - 1][n - 1] = 3 - x[n - 1]
        ret[n - 1][n - 2] = -1
        for i in range(1, n - 1):
            ret[i][i - 1] = -1
            ret[i][i] = 3 - x[i]
            ret[i][i + 1] = -2
        return ret

    N = 4

    def newton_solve():
        x_start = np.random.uniform(low=-1, high=1, size=N)
        x_start = []
        for i in range(N // 2):
            x_start.append(1)

        for i in range(N // 2):
            x_start.append(-1)

        x = Solver(f, x_start, f_prime, max_iter=2000, debug=True).newton()
        print(f'usual diff {distance(np.zeros(N), f(x))}')
        print(f'root {x}')

    def broyden_solve():
        # x_start = np.random.uniform(low=-1, high=1, size=N)
        x_start = []
        for i in range(N // 2):
            x_start.append(1)
        for i in range(N // 2):
            x_start.append(-1)

        x = Solver(f, x_start, f_prime, max_iter=2000, debug=True).broyden()
        print(f'usual diff {distance(np.zeros(N), f(x))}')
        print(f'root {x}')

    def interval_solve():
        x_start = [interval([-1.5, 1.5]) for i in range(N)]
        x = IntervalSolver(f_interval, x_start, f_interval_prime, max_iter=2000, debug=True).krawczyk()
        print(f'interval diff {distance(np.zeros(N), f_interval(x))}')

    print('Newton:')
    start = time.time()
    newton_solve()
    end = time.time()
    print(f'Elapsed time : {round(end - start, 3)}')
    print('—' * 100)
    print('Broyden:')
    start = time.time()
    broyden_solve()
    end = time.time()
    print(f'Elapsed time : {round(end - start, 3)}')
    print('—' * 100)
    print("krawczyk:")
    start = time.time()
    interval_solve()
    end = time.time()
    print(f'Elapsed time : {round(end - start, 3)}')
    print('—' * 100)


def test_big_exp():
    def f(x):
        n = len(x)
        assert n > 3
        ret = [0] * n
        ret[0] = math.exp(x[0]) - 1
        for i in range(1, n):
            ret[i] = (i + 1) / 10.0 * (math.exp(x[i]) + x[i - 1] - 1)
        return ret

    def f_interval(x):
        n = len(x)
        assert n > 3
        ret = [interval(0, 0)] * n
        ret[0] = imath.exp(x[0]) - 1
        for i in range(1, n):
            ret[i] = (i + 1) / 10.0 * (imath.exp(x[i]) + x[i - 1] - 1)
        return ret

    def f_prime(x):
        n = len(x)
        assert n > 3
        ret = [[0] * n for i in range(n)]
        ret[0][0] = math.exp(x[0])
        for i in range(1, n):
            ret[i][i - 1] = (i + 1) / 10.0
            ret[i][i] = (i + 1) / 10.0 * math.exp(x[i])
        return ret

    def f_prime_interval(x):
        n = len(x)
        assert n > 3
        ret = [[0] * n for i in range(n)]
        ret[0][0] = imath.exp(x[0])
        for i in range(1, n):
            ret[i][i - 1] = (i + 1) / 10.0
            ret[i][i] = (i + 1) / 10.0 * imath.exp(x[i])
        return ret

    N = 10

    def newton_solve():
        x_start = []
        for i in range(N // 2):
            x_start.append(-0.5)
        for i in range(N // 2):
            x_start.append(0.5)
        # x_start = [1] * N

        x = Solver(f, x_start, f_prime, max_iter=1000, debug=True).newton()
        print(f'newton diff {distance(np.zeros(N), f(x))}')
        print(f'root {x}')

    def broyden_solve():
        x_start = []
        for i in range(N // 2):
            x_start.append(-0.5)
        for i in range(N // 2):
            x_start.append(0.5)
        x = Solver(f, x_start, f_prime, max_iter=1000, debug=True).broyden()
        print(f'broyden diff {distance(np.zeros(N), f(x))}')
        print(f'root {x}')

    def interval_solve():
        x_start = [interval([-1.5, 1]) for i in range(N)]
        x = IntervalSolver(f_interval, x_start, f_prime_interval, max_iter=100, debug=True).krawczyk()
        # print(x, f_interval(x))
        print(f'interval diff {distance(np.zeros(N), f_interval(x))}')

    print('Newton:')
    start = time.time()
    newton_solve()
    end = time.time()
    print(f'Elapsed time : {round(end - start, 3)}')
    print('—' * 100)

    print('Broyden:')
    start = time.time()
    broyden_solve()
    end = time.time()
    print(f'Elapsed time : {round(end - start, 3)}')
    print('—' * 100)

    print("krawczyk:")
    start = time.time()
    interval_solve()
    end = time.time()
    print(f'Elapsed time : {round(end - start, 3)}')
    print('—' * 100)




# test_big_exp()
# test_broyden_tridiagonal()


# Ниже функции на которых тестится базовый функционал и проверка работоспособности кода солверов

def test_simple():
    def f(x):
        # x − y + 1 = 0
        # x^2 - y + 1 = 0
        return [x[0] - x[1] + 1, x[0] ** 2 - x[1] + 1]

    def f_prime(x):
        return [[1, -1],
                [2 * x[0], -1]]

    root = Solver(f, (1, 1), f_prime, debug=True).newton()
    # root = Solver(f, (1, 1), f_prime, debug=True).broyden()
    root = IntervalSolver(f, [interval([-1, 1]), interval([-1, 1])], f_prime, debug=True).krawczyk()

    print("root is : ", root[0], root[1], '; Value of function in root :', f(root))


def test_undetermined_simple():
    def f(x):
        # x − y + 1 + z = 0
        # x^2 - y + 1 + z= 0
        # 0 = 0
        return [x[0] - x[1] + 1 + x[2], x[0] ** 2 - x[1] + 1 + x[2], 0]

    def f_prime(x):
        return [[1, -1, 1],
                [2 * x[0], -1, 1],
                [0, 0, 0]]

    root = Solver(f, (1, 1, 1), f_prime, debug=True).newton()
    # root = Solver(f, (1, 1, 1), f_prime, debug=True).broyden()
    root = IntervalSolver(f, [interval([-2, 2]), interval([-2, 2]), interval([-2, 2])], f_prime, debug=True).krawczyk()

    print("root of function from mine solver : ", root[0], root[1], root[2], '; Value of function in root :', f(root))


def test_another_simple():
    def f(x):
        # 3x1 - cos(x2x3)-1/2=0
        # x1^2 - 81(x2+0.1)^2+sin(x3)+1.06=0
        # e^{-x1x2}+20x3+(10pi-3)/3=0
        # sols : (0.5, 0, -0.52)
        return [3 * x[0] - math.cos(x[1] * x[2]) - 1 / 2,
                x[0] ** 2 - 81 * (x[1] + 0.1) ** 2 + math.sin(x[2]) + 1.06,
                math.exp(-x[0] * x[1]) + 20 * x[2] + (10 * math.pi - 3) / 3]

    def f_prime(x):
        return [[3, x[2] * math.sin(x[1] * x[2]), x[1] * math.sin(x[1] * x[2])],
                [2 * x[0], -162 * (x[1] + 0.1), math.cos(x[2])],
                [-x[1] * math.exp(-x[0] * x[1]), -x[0] * math.exp(-x[0] * x[1]), 20]]

    root = opt.fsolve(f, (0.1, 0.1, -0.1), args=(), fprime=f_prime)

    print("root of function from fsolve : ", root[0], root[1], root[2], '; Value of function in root :', f(root))

    root = Solver(f, (0.1, 0.1, -0.1), f_prime).newton()

    print("root of function from mine solver (newton) : ", root[0], root[1], root[2], '; Value of function in root :',
          f(root))

    root = Solver(f, (0.1, 0.1, -0.1), f_prime).broyden()

    print("root of function from mine solver (broyden) : ", root[0], root[1], root[2], '; Value of function in root :',
          f(root))


def test_interval_small():
    def f(x):
        return [x[0] ** 2 + x[1] ** 2 - 1, x[0] - x[1] ** 2]

    def f_prime(x):
        return [[2 * x[0], 2 * x[1]], [1, -2 * x[1]]]

    x = Solver(f, (10, 10), f_prime).newton()

    print("f mine : ", x, f(x))

    print('─' * 100)

    x = Solver(f, (10, 10), f_prime).broyden()

    print("f mine : ", x, f(x))

    print('─' * 100)

    # x = IntervalSolver(f, (interval([0.5, 0.8]), interval([0.6, 0.9])), f_prime).krawczyk()
    # ans = [x[i][0].inf for i in range(len(x))]
    # print("f mine : ", ans, f(ans))
    #
    # print('─' * 100)

    x = IntervalSolver(f, (interval([0.0, 1.0]), interval([0.0, 1.0])), f_prime).krawczyk()
    ans = [x[i][0].inf for i in range(len(x))]
    print("f mine : ", ans, f(ans))

    print('─' * 100)


def test_rosenbrock():
    def f(x):
        n = len(x)
        assert n > 3
        ret = [0] * n

        for i in range(n // 2):
            ret[2 * i] = 10 * (x[2 * i + 1] - x[2 * i] ** 2)
            ret[2 * i + 1] = 1 - x[2 * i]
        return ret

    def f_prime(x):
        n = len(x)
        assert n > 3
        ret = [[0] * n for i in range(n)]
        k = n // 2
        for i in range(0, k):
            ret[2 * i][2 * i + 1] = 10
            ret[2 * i][2 * i] = -20 * x[2 * i]
            ret[2 * i + 1][2 * i] = -1

        return ret

    N = 10

    assert N % 2 == 0

    def newton_solve():
        x_start = np.random.uniform(low=0, high=N, size=N)
        x = Solver(f, x_start, f_prime, max_iter=1000, debug=True).broyden()
        print(f'newton diff {distance(np.zeros(N), f(x))}')

    def broyden_solve():
        x_start = np.random.uniform(low=0, high=N, size=N)
        x = Solver(f, x_start, f_prime, max_iter=1000, debug=True).broyden()
        print(f'broyden diff {distance(np.zeros(N), f(x))}')

    def interval_solve():
        x_start = [interval([-N, N]) for i in range(N)]
        x = IntervalSolver(f, x_start, f_prime, max_iter=1000, debug=True).krawczyk()
        print(f'interval diff {distance(np.zeros(N), f(x))}, \nroot is {x}')

    newton_solve()
    broyden_solve()
    interval_solve()
