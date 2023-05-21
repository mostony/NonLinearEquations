import numpy as np
from interval import interval, inf, imath


def distance(x1, x2):
    """
    Caclulate the distance between two points in Euclidian metric
    :param x1: first point
    :param x2:
    :return:
    """
    if len(x1) != len(x2):
        raise RuntimeError('Points in different spaces - number of coordinates is different')
    ret = 0
    for i in range(0, len(x1)):
        ret += (x1[i] - x2[i]) ** 2
    return ret ** 0.5


class Solver:
    def __init__(self, f, x0, f_deriv=None, x_eps=1e-8, max_iter=100, debug=True):
        """
        :param f: set of functions - m functions on n variables
        :param x0: starting point for finding roots. Point has n coordinates
        :param f_deriv: Jacobin of functions
        :param x_eps: The search will stop if the difference between
        two consecutive iterates is at most xtol
        :param max_iter: Max number of iterations
        """
        self.f = f
        self.f_deriv = f_deriv
        self.x0 = x0
        self.n = len(self.x0)
        self.x_eps = x_eps
        self.tol = x_eps
        self.max_iter = max_iter
        self.debug = debug

    # def Run(self):
    #     return self.__newton__()

        # if different sign, then bisection method
        # if self.interval[0] * self.interval[1] < 0:
        #     return self.__bisection__(self)
        # pass

    def newton(self):
        if self.f_deriv == None:
            raise RuntimeError('Empty Jacobian')
        cur_x = self.x0
        for it in range(self.max_iter):
            if self.debug:
                print(f'On {it} iteration current vector is {cur_x}')
            J = np.array(self.f_deriv(cur_x))
            F = np.array(self.f(cur_x))
            if np.linalg.det(J) == 0:
                raise RuntimeError('Determinant equals zero')
            cur_y = np.linalg.solve(J, -F)
            next_x = cur_x + cur_y
            if distance(cur_x, next_x) < self.x_eps:
                return cur_x
            cur_x = next_x
        return cur_x

    def broyden(self):
        if self.f_deriv == None:
            raise RuntimeError('Empty Jacobian')
        cur_x = np.array(self.x0).reshape((self.n, 1))
        J = np.array(self.f_deriv(list(cur_x.ravel())))
        if np.linalg.det(J) == 0:
            raise RuntimeError('Determinant equals zero')
        A_cur = np.linalg.inv(J)

        for it in range(self.max_iter):
            if self.debug:
                print(f'On {it} iteration current vector is {list(cur_x.ravel())}')
            f_value = np.array(self.f(list(cur_x.ravel()))).reshape((self.n, 1))
            next_x = cur_x - A_cur @ f_value
            y = np.array(self.f(list(next_x.ravel()))) - np.array(self.f(cur_x.ravel()))
            y = y.reshape((self.n, 1))
            s = np.array(next_x - cur_x).reshape((self.n, 1))
            t = s.T @ A_cur @ y
            A_next = A_cur + 1 / t * ((s - A_cur @ y) @ s.T @ A_cur)
            A_cur = A_next
            if distance(cur_x, next_x) < self.x_eps:
                return list(cur_x.ravel())
            cur_x = next_x
        return list(cur_x.ravel())


class IntervalVector:
    def __init__(self, size):
        self.elems = [interval(0, 0)] * size
        self.size = size

    def __repr__(self):
        return repr(self.elems)

    def __mul__(self, other):
        if self.size != other.size:
            raise ArithmeticError("vectors of two different lengths")
        a = interval([0, 0])
        for i in range(self.size):
            a += self.elems[i] * other.elems[i]
        return a

    def set(self, array):
        for i in range(self.size):
            self.elems[i] = array[i]


# class IntervalMatrix:
#     def __init__(self, n):
#         self.col = [IntervalVector(c) for i in range(r)]
#         self.n = n
#         self.r = r
#         self.c = c
#
#     def __repr__(self):
#         return repr(self.col)
#
#     def __mul__(self, other):
#         if type(other) != IntervalVector:
#             raise TypeError("matrices can only be multiplied by vectors")
#
#         if self.c != other.size:
#             raise ArithmeticError("rows and lengths do not match")
#         a = IntervalVector(self.r)
#         a.set([(other * self.coloums[i]) for i in range(self.r)])
#         return a
#
#     def set(self, multiarray):
#         for i in range(self.c):
#             self.coloums[i].set(multiarray[i])


class IntervalSolver:

    def __init__(self, f, X: interval, f_deriv=None, x_eps=1e-8, max_iter=100, debug=True):
        """
        :param f: set of functions - m functions on n variables
        :param x0: starting point for finding roots. Point has n coordinates
        :param f_deriv: Jacobin of functions
        :param x_eps: The search will stop if the difference between
        two consecutive iterates is at most xtol
        :param max_iter: Max number of iterations
        """
        self.f = f
        self.f_deriv = f_deriv
        self.X = X
        self.n = len(self.X)
        self.x_eps = x_eps
        self.tol = x_eps
        self.max_iter = max_iter
        self.debug = debug

    def krawczyk(self):
        if self.f_deriv is None:
            raise RuntimeError('Empty Jacobian')

        current_interval = list(self.X)
        for it in range(self.max_iter):
            if self.debug:
                print(f'On {it} iteration current vector is {current_interval}')

            y = [interv.midpoint for interv in current_interval]
            Z = [current_interval[i] - y[i] for i in range(self.n)]
            tmp = self.f_deriv(current_interval)
            Y = np.zeros((self.n, self.n))
            for i in range(self.n):
                for j in range(self.n):
                    tmp[i][j] = interval(tmp[i][j])
                    Y[i][j] = tmp[i][j].midpoint[0].inf
            Y = np.linalg.inv(Y)

            f_val = self.f(y)
            next_interval = y.copy()

            for i in range(self.n):
                for j in range(self.n):
                    next_interval[i] -= interval(Y[i][j]) * f_val[j]

            C = [[0] * self.n] * self.n

            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n):
                        C[i][j] -= interval(Y[i][k]) * tmp[k][j]

            for i in range(self.n):
                C[i][i] += 1

            mda = [interval(0, 0)] * self.n
            for i in range(self.n):
                for j in range(self.n):
                    mda[i] += C[i][j] * Z[j]

            for i in range(self.n):
                next_interval[i] += mda[i]
                next_interval[i] = interval(next_interval[i])

            for i in range(self.n):
                current_interval[i] &= next_interval[i]

            xx = [0] * self.n
            for i in range(0, self.n):
                interv = interval(current_interval[i])[0]
                xx[i] = interv.sup - interv.inf
            if distance([0] * self.n, xx) < self.x_eps:
                break
        return current_interval
