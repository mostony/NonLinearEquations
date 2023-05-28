import numpy as np
from interval import interval


def distance(x1, x2):
    """
    Caclulate the distance between two points in Euclidian metric
    :param x1: first point
    :param x2: second point
    :return: ||x1 - x2||_2
    """
    if len(x1) != len(x2):
        raise RuntimeError('Points in different spaces - number of coordinates is different')
    ret = 0
    for i in range(0, len(x1)):
        ret += (x1[i] - x2[i]) ** 2
    return ret ** 0.5


class Solver:
    def __init__(self, f, x0, f_deriv=None, x_eps=1e-8, y_eps=1e-8, max_iter=1000, debug=False):
        """
        :param f: set of functions - m functions on n variables
        :param x0: starting point for finding roots. Point has n coordinates
        :param f_deriv: Jacobin of functions
        :param x_eps: The search will stop if the difference between
        two consecutive iterates is at most x_eps
        :param y_eps: stop if ||f(x)|| < y_eps
        :param max_iter: Max number of iterations
        :param debug: if debug is True, then print current approximation after each iteration
        """
        self.f = f
        self.f_deriv = f_deriv
        self.x0 = x0
        self.n = len(self.x0)
        self.x_eps = x_eps
        self.y_eps = y_eps
        self.max_iter = max_iter
        self.debug = debug

    def newton(self):
        """
        Caclulate root of equations using newton's method
        :return: root of equations
        """
        if self.f_deriv is None:
            raise RuntimeError('Empty Jacobian')

        cur_x = self.x0
        for it in range(self.max_iter):
            if self.debug:
                print(f'On {it} iteration current vector is {cur_x}')

            J = np.array(self.f_deriv(cur_x))
            F = np.array(self.f(cur_x))

            cur_y = np.zeros(1)

            # if matrix is non invertable then take pseudo - inverse
            if J.shape[0] != J.shape[1] or np.linalg.det(J) == 0:
                tmp = np.linalg.pinv(J)
                cur_y = tmp @ -F
            else:
                cur_y = np.linalg.solve(J, -F)

            next_x = cur_x + cur_y
            if distance(cur_x, next_x) < self.x_eps:
                return cur_x
            if distance(self.f(cur_x), np.zeros(self.n)) < self.y_eps:
                return cur_x
            cur_x = next_x
        return cur_x

    def broyden(self):
        """
        Caclulate root of equations using broyden's method, if Jacobian exists and invertable
        :return: root of equations
        """
        if self.f_deriv is None:
            raise RuntimeError('Empty Jacobian')

        cur_x = np.array(self.x0).reshape((self.n, 1))
        J = np.array(self.f_deriv(list(cur_x.ravel())))
        A_cur = np.zeros(1)

        # if matrix is non invertable then take pseudo - inverse
        if J.shape[0] != J.shape[1] or np.linalg.det(J) == 0:
            A_cur = np.linalg.pinv(J)
        else:
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
            if distance(self.f(cur_x), np.zeros(self.n)) < self.y_eps:
                return cur_x
            cur_x = next_x
        return list(cur_x.ravel())


class IntervalSolver:

    def __init__(self, f, X: interval, f_deriv=None, x_eps=1e-8, y_eps=1e-8, max_iter=100, debug=True):
        """
        :param f: set of functions - m functions on n variables
        :param x0: starting interval for finding roots. Interval has n coordinates
        :param f_deriv: Jacobin of functions
        :param x_eps: The search will stop if the diagonal of the box becomes smaller than x_eps
        :param y_eps: stop if ||f(x)|| < y_eps
        :param max_iter: Max number of iterations
        :param degub: if debug is True, then print interval after each iteration
        """
        self.f = f
        self.f_deriv = f_deriv
        self.X = X
        self.n = len(self.X)
        self.x_eps = x_eps
        self.y_eps = y_eps
        self.max_iter = max_iter
        self.debug = debug
        self.gap = 1e-5
        # gap - apply bisection to the widest interval if new_x close to cur_x

    def krawczyk(self):
        """
        Caclulate root of equations using broyden's method, if Jacobian exists and invertable
        :return: root of equations

        """
        if self.f_deriv is None:
            raise RuntimeError('Empty Jacobian')

        current_interval = list(self.X)

        for it in range(self.max_iter):

            if self.debug == True:
                print(f'On {it} iteration current interval is {current_interval}')

            y = [interv.midpoint[0].inf for interv in current_interval]

            Z = [current_interval[i] - y[i] for i in range(self.n)]

            J_approx = self.f_deriv(current_interval)
            J_mid_inv = np.zeros((self.n, self.n))
            for i in range(self.n):
                for j in range(self.n):
                    J_approx[i][j] = interval(J_approx[i][j])
                    J_mid_inv[i][j] = J_approx[i][j].midpoint[0].inf

            J_mid_inv = np.linalg.pinv(J_mid_inv)

            f_val = self.f(y)
            next_interval = y.copy()

            for i in range(self.n):
                for j in range(self.n):
                    next_interval[i] -= interval(J_mid_inv[i][j]) * f_val[j]
                    next_interval[i] = interval(next_interval[i])

            C = [[interval(0, 0) for x in range(self.n)] for y in range(self.n)]

            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n):
                        C[i][j] -= interval(J_mid_inv[i][k]) * J_approx[k][j]

            for i in range(self.n):
                C[i][i] += 1

            mda = [interval(0, 0)] * self.n
            for i in range(self.n):
                for j in range(self.n):
                    mda[i] += C[i][j] * Z[j]

            for i in range(self.n):
                next_interval[i] += mda[i]

            prev_interval = current_interval.copy()
            for i in range(self.n):
                current_interval[i] &= next_interval[i]

            int_dist = 0
            for i in range(self.n):
                l1 = current_interval[i][0].inf
                r1 = current_interval[i][0].sup
                l2 = prev_interval[i][0].inf
                r2 = prev_interval[i][0].sup
                d = r2 - l2 - (r1 - l1)
                int_dist = max(d, int_dist)
            int_dist **= 0.5

            if int_dist < self.gap:
                pos_change = 0
                best_w = 0
                for i in range(self.n):
                    l = current_interval[i][0].inf
                    r = current_interval[i][0].sup
                    if r - l > best_w:
                        best_w = r - l
                        pos_change = i

                interv = current_interval[pos_change][0]
                l = interv.inf
                r = interv.sup
                current_interval[pos_change] = interval([l, (l + r) / 2])
                t = self.f(current_interval)
                current_interval[pos_change] = interval([(l + r) / 2, r])

                for i in range(self.n):
                    if 0 in t[i]:
                        pass
                    else:
                        current_interval[pos_change] = interval([(l + r) / 2, r])
                        break

                continue

            xx = [0] * self.n
            for i in range(0, self.n):
                if len(current_interval[i]) == 0:
                    print(self.f(prev_interval))
                    raise RuntimeError('No solution')
                interv = interval(current_interval[i])[0]
                xx[i] = float(interv.sup - interv.inf)

            if distance([0] * self.n, xx) < self.x_eps:
                return current_interval

        return current_interval
