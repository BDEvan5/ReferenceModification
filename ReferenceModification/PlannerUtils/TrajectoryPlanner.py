from casadi.casadi import PrintableCommon, fmax
import numpy as np
import casadi as ca 
from matplotlib import pyplot as plt 

import toy_auto_race.Utils.LibFunctions as lib 

def MinCurvatureTrajectory(pts, nvecs, ws):
    """
    This function uses optimisation to minimise the curvature of the path
    """
    # w_min = ws[:, 0] 
    w_min = - ws[:, 0] 
    w_max = ws[:, 1] 
    th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]

    N = len(pts)

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)
    th_f = ca.MX.sym('n_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)
    th1_f = ca.MX.sym('y1_f', N-1)
    th2_f = ca.MX.sym('y1_f', N-1)
    th1_f1 = ca.MX.sym('y1_f', N-2)
    th2_f1 = ca.MX.sym('y1_f', N-2)

    o_x_s = ca.Function('o_x', [n_f], [pts[:-1, 0] + nvecs[:-1, 0] * n_f])
    o_y_s = ca.Function('o_y', [n_f], [pts[:-1, 1] + nvecs[:-1, 1] * n_f])
    o_x_e = ca.Function('o_x', [n_f], [pts[1:, 0] + nvecs[1:, 0] * n_f])
    o_y_e = ca.Function('o_y', [n_f], [pts[1:, 1] + nvecs[1:, 1] * n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

    real = ca.Function('real', [th1_f, th2_f], [ca.cos(th1_f)*ca.cos(th2_f) + ca.sin(th1_f)*ca.sin(th2_f)])
    im = ca.Function('im', [th1_f, th2_f], [-ca.cos(th1_f)*ca.sin(th2_f) + ca.sin(th1_f)*ca.cos(th2_f)])

    sub_cmplx = ca.Function('a_cpx', [th1_f, th2_f], [ca.atan2(im(th1_f, th2_f),real(th1_f, th2_f))])
    
    get_th_n = ca.Function('gth', [th_f], [sub_cmplx(ca.pi*np.ones(N-1), sub_cmplx(th_f, th_ns[:-1]))])
    d_n = ca.Function('d_n', [n_f_a, th_f], [track_length(n_f_a)/ca.tan(get_th_n(th_f))])

    # objective
    real1 = ca.Function('real1', [th1_f1, th2_f1], [ca.cos(th1_f1)*ca.cos(th2_f1) + ca.sin(th1_f1)*ca.sin(th2_f1)])
    im1 = ca.Function('im1', [th1_f1, th2_f1], [-ca.cos(th1_f1)*ca.sin(th2_f1) + ca.sin(th1_f1)*ca.cos(th2_f1)])

    sub_cmplx1 = ca.Function('a_cpx1', [th1_f1, th2_f1], [ca.atan2(im1(th1_f1, th2_f1),real1(th1_f1, th2_f1))])
    
    # define symbols
    n = ca.MX.sym('n', N)
    th = ca.MX.sym('th', N-1)

    nlp = {\
    'x': ca.vertcat(n, th),
    'f': ca.sumsqr(sub_cmplx1(th[1:], th[:-1])), 
    # 'f': ca.sumsqr(track_length(n)), 
    'g': ca.vertcat(
                # dynamic constraints
                n[1:] - (n[:-1] + d_n(n, th)),

                # boundary constraints
                n[0], #th[0],
                n[-1], #th[-1],
            ) \
    
    }

    # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})
    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})

    ones = np.ones(N)
    n0 = ones*0

    th0 = []
    for i in range(N-1):
        th_00 = lib.get_bearing(pts[i, 0:2], pts[i+1, 0:2])
        th0.append(th_00)

    th0 = np.array(th0)

    x0 = ca.vertcat(n0, th0)

    lbx = list(w_min) + [-np.pi]*(N-1) 
    ubx = list(w_max) + [np.pi]*(N-1) 

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    x_opt = r['x']
    print(f"Sol Status (Optimal found?): {S.stats()['success']}")

    n_set = np.array(x_opt[:N])
    # thetas = np.array(x_opt[1*N:2*(N-1)])

    return n_set


def MinCurvatureTrajectoryForest(pts, ws):
    """
    This function uses optimisation to minimise the curvature of the path
    """
    w_min = - ws[:, 0] * 0.9
    w_max = ws[:, 1] * 0.9
    N = len(pts)
    ys = pts[:, 1] # constant
    th_ns = [np.pi for i in range(N)]

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)
    th_f = ca.MX.sym('n_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)
    th1_f = ca.MX.sym('y1_f', N-1)
    th2_f = ca.MX.sym('y1_f', N-1)
    th1_f1 = ca.MX.sym('y1_f', N-2)
    th2_f1 = ca.MX.sym('y1_f', N-2)

    o_x_s = ca.Function('o_x', [n_f], [pts[:-1, 0] + n_f])
    o_x_e = ca.Function('o_x', [n_f], [pts[1:, 0] +  n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                ys[:-1], ys[1:])])

    real = ca.Function('real', [th1_f, th2_f], [ca.cos(th1_f)*ca.cos(th2_f) + ca.sin(th1_f)*ca.sin(th2_f)])
    im = ca.Function('im', [th1_f, th2_f], [-ca.cos(th1_f)*ca.sin(th2_f) + ca.sin(th1_f)*ca.cos(th2_f)])

    sub_cmplx = ca.Function('a_cpx', [th1_f, th2_f], [ca.atan2(im(th1_f, th2_f),real(th1_f, th2_f))])
    
    get_th_n = ca.Function('gth', [th_f], [sub_cmplx(ca.pi*np.ones(N-1), sub_cmplx(th_f, th_ns[:-1]))])
    d_n = ca.Function('d_n', [n_f_a, th_f], [track_length(n_f_a)/ca.tan(get_th_n(th_f))])

    # objective
    real1 = ca.Function('real1', [th1_f1, th2_f1], [ca.cos(th1_f1)*ca.cos(th2_f1) + ca.sin(th1_f1)*ca.sin(th2_f1)])
    im1 = ca.Function('im1', [th1_f1, th2_f1], [-ca.cos(th1_f1)*ca.sin(th2_f1) + ca.sin(th1_f1)*ca.cos(th2_f1)])

    sub_cmplx1 = ca.Function('a_cpx1', [th1_f1, th2_f1], [ca.atan2(im1(th1_f1, th2_f1),real1(th1_f1, th2_f1))])
    
    # define symbols
    n = ca.MX.sym('n', N)
    th = ca.MX.sym('th', N-1)

    nlp = {\
    'x': ca.vertcat(n, th),
    'f': ca.sumsqr(sub_cmplx1(th[1:], th[:-1])), 
    # 'f': ca.sumsqr(track_length(n)), 
    'g': ca.vertcat(
                # dynamic constraints
                n[1:] - (n[:-1] + d_n(n, th)),

                # boundary constraints
                n[0], #th[0],
                n[-1], #th[-1],
            ) \
    
    }

    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})
    # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})

    ones = np.ones(N)
    n0 = ones*0

    th0 = []
    for i in range(N-1):
        th_00 = lib.get_bearing(pts[i, 0:2], pts[i+1, 0:2])
        th0.append(th_00)

    th0 = np.array(th0)

    x0 = ca.vertcat(n0, th0)

    lbx = list(w_min) + [-np.pi]*(N-1) 
    ubx = list(w_max) + [np.pi]*(N-1) 

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    x_opt = r['x']
    print(f"Sol Status (Optimal found?): {S.stats()['success']}")

    n_set = np.array(x_opt[:N])
    # thetas = np.array(x_opt[1*N:2*(N-1)])

    return n_set


def find_true_widths(pts, nvecs, ws, check_scan_location):
    tx = pts[:, 0]
    ty = pts[:, 1]
    onws = ws[:, 0]
    opws = ws[:, 1]

    stp_sze = 0.1
    sf = 0.5 # safety factor
    N = len(pts)
    nws, pws = [], []
    for i in range(N):
        pt = [tx[i], ty[i]]
        nvec = nvecs[i]

        if not check_scan_location(pt):
            j = stp_sze
            s_pt = lib.add_locations(pt, nvec, j)
            while not check_scan_location(s_pt) and j < opws[i]:
                j += stp_sze
                s_pt = lib.add_locations(pt, nvec, j)
            pws.append(j*sf)

            j = stp_sze
            s_pt = lib.sub_locations(pt, nvec, j)
            while not check_scan_location(s_pt) and j < onws[i]:
                j += stp_sze
                s_pt = lib.sub_locations(pt, nvec, j)
            nws.append(j*sf)
            # print(f"Pt added without being adjusted")
        else:
            print(f"Obs in way of pt: {i}")

            for j in np.linspace(0, onws[i], 10):
                p_pt = lib.add_locations(pt, nvec, j)
                n_pt = lib.sub_locations(pt, nvec, j)
                if not check_scan_location(p_pt):
                    nws.append(-j*(1+sf))
                    pws.append(opws[i])
                    print(f"PosPt NewW: [{-j*(1+sf)}, {opws[i]}]")
                    break
                elif not check_scan_location(n_pt):
                    pws.append(-j*(1+sf))
                    nws.append(onws[i])
                    print(f"PosPt NewW: [{-j*(1+sf)}, {onws[i]}]")
                    break 
                if j == onws[i]:
                    print(f"Problem - no space found")


    nws, pws = np.array(nws), np.array(pws)
    ws = np.concatenate([nws[:, None], pws[:, None]], axis=-1)

    return ws




def ObsAvoidTraj(pts, nvecs, ws, check_scan_location):
    ws = find_true_widths(pts, nvecs, ws, check_scan_location)

    return MinCurvatureTrajectory(pts, nvecs, ws)


def ShortestTraj(pts, nvecs, ws, check_scan_location):
    ws = find_true_widths(pts, nvecs, ws, check_scan_location)

    w_min = - ws[:, 0] * 0.9
    w_max = ws[:, 1] * 0.9
    th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]

    N = len(pts)

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)
    th_f = ca.MX.sym('n_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)
    th1_f = ca.MX.sym('y1_f', N-1)
    th2_f = ca.MX.sym('y1_f', N-1)
    th1_f1 = ca.MX.sym('y1_f', N-2)
    th2_f1 = ca.MX.sym('y1_f', N-2)

    o_x_s = ca.Function('o_x', [n_f], [pts[:-1, 0] + nvecs[:-1, 0] * n_f])
    o_y_s = ca.Function('o_y', [n_f], [pts[:-1, 1] + nvecs[:-1, 1] * n_f])
    o_x_e = ca.Function('o_x', [n_f], [pts[1:, 0] + nvecs[1:, 0] * n_f])
    o_y_e = ca.Function('o_y', [n_f], [pts[1:, 1] + nvecs[1:, 1] * n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

    real = ca.Function('real', [th1_f, th2_f], [ca.cos(th1_f)*ca.cos(th2_f) + ca.sin(th1_f)*ca.sin(th2_f)])
    im = ca.Function('im', [th1_f, th2_f], [-ca.cos(th1_f)*ca.sin(th2_f) + ca.sin(th1_f)*ca.cos(th2_f)])

    sub_cmplx = ca.Function('a_cpx', [th1_f, th2_f], [ca.atan2(im(th1_f, th2_f),real(th1_f, th2_f))])
    
    get_th_n = ca.Function('gth', [th_f], [sub_cmplx(ca.pi*np.ones(N-1), sub_cmplx(th_f, th_ns[:-1]))])
    d_n = ca.Function('d_n', [n_f_a, th_f], [track_length(n_f_a)/ca.tan(get_th_n(th_f))])

    # objective
    real1 = ca.Function('real1', [th1_f1, th2_f1], [ca.cos(th1_f1)*ca.cos(th2_f1) + ca.sin(th1_f1)*ca.sin(th2_f1)])
    im1 = ca.Function('im1', [th1_f1, th2_f1], [-ca.cos(th1_f1)*ca.sin(th2_f1) + ca.sin(th1_f1)*ca.cos(th2_f1)])

    sub_cmplx1 = ca.Function('a_cpx1', [th1_f1, th2_f1], [ca.atan2(im1(th1_f1, th2_f1),real1(th1_f1, th2_f1))])
    
    # define symbols
    n = ca.MX.sym('n', N)
    th = ca.MX.sym('th', N-1)

    nlp = {\
    'x': ca.vertcat(n, th),
    # 'f': ca.sumsqr(sub_cmplx1(th[1:], th[:-1])) * 5 + ca.sumsqr(track_length(n)), 
    'f': ca.sumsqr(track_length(n)), 
    'g': ca.vertcat(
                # dynamic constraints
                n[1:] - (n[:-1] + d_n(n, th)),

                # boundary constraints
                n[0], #th[0],
                n[-1], #th[-1],
            ) \
    
    }

    # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})
    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})

    ones = np.ones(N)
    n0 = ones*0

    th0 = []
    for i in range(N-1):
        th_00 = lib.get_bearing(pts[i, 0:2], pts[i+1, 0:2])
        th0.append(th_00)

    th0 = np.array(th0)

    x0 = ca.vertcat(n0, th0)

    lbx = list(w_min) + [-np.pi]*(N-1) 
    ubx = list(w_max) + [np.pi]*(N-1) 

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    x_opt = r['x']

    n_set = np.array(x_opt[:N])
    thetas = np.array(x_opt[1*N:2*(N-1)])


    return n_set


"""Find the max velocity """
def Max_velocity(pts, config, show=False):
    mu = config['car']['mu']
    m = config['car']['m']
    g = config['car']['g']
    l_f = config['car']['l_f']
    l_r = config['car']['l_r']
    safety_f = config['pp']['force_f']
    f_max = mu * m * g * safety_f
    f_long_max = l_f / (l_r + l_f) * f_max
    max_v = config['lims']['max_v']  # parameter to be adapted so that optimiser isnt too fast
    max_a = config['lims']['max_a']

    s_i, th_i = convert_pts_s_th(pts)
    th_i_1 = th_i[:-1]
    s_i_1 = s_i[:-1]
    N = len(s_i)
    N1 = len(s_i) - 1

    # setup possible casadi functions
    d_x = ca.MX.sym('d_x', N-1)
    d_y = ca.MX.sym('d_y', N-1)
    vel = ca.Function('vel', [d_x, d_y], [ca.sqrt(ca.power(d_x, 2) + ca.power(d_y, 2))])

    dx = ca.MX.sym('dx', N)
    dy = ca.MX.sym('dy', N)
    dt = ca.MX.sym('t', N-1)
    f_long = ca.MX.sym('f_long', N-1)
    f_lat = ca.MX.sym('f_lat', N-1)

    nlp = {\
        'x': ca.vertcat(dx, dy, dt, f_long, f_lat),
        'f': ca.sum1(dt), 
        'g': ca.vertcat(
                    # dynamic constraints
                    dt - s_i_1 / ((vel(dx[:-1], dy[:-1]) + vel(dx[1:], dy[1:])) / 2 ),
                    # ca.arctan2(dy, dx) - th_i,
                    dx/dy - ca.tan(th_i),
                    dx[1:] - (dx[:-1] + (ca.sin(th_i_1) * f_long + ca.cos(th_i_1) * f_lat) * dt  / m),
                    dy[1:] - (dy[:-1] + (ca.cos(th_i_1) * f_long - ca.sin(th_i_1) * f_lat) * dt  / m),

                    # path constraints
                    ca.sqrt(ca.power(f_long, 2) + ca.power(f_lat, 2)),

                    # boundary constraints
                    # dx[0], dy[0]
                ) \
    }

    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})
    # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})

    # make init sol
    v0 = np.ones(N) * max_v/2
    dx0 = v0 * np.sin(th_i)
    dy0 = v0 * np.cos(th_i)
    dt0 = s_i_1 / ca.sqrt(ca.power(dx0[:-1], 2) + ca.power(dy0[:-1], 2)) 
    f_long0 = np.zeros(N-1)
    ddx0 = dx0[1:] - dx0[:-1]
    ddy0 = dy0[1:] - dy0[:-1]
    a0 = (ddx0**2 + ddy0**2)**0.5 
    f_lat0 = a0 * m

    x0 = ca.vertcat(dx0, dy0, dt0, f_long0, f_lat0)

    # make lbx, ubx
    # lbx = [-max_v] * N + [-max_v] * N + [0] * N1 + [-f_long_max] * N1 + [-f_max] * N1
    lbx = [-max_v] * N + [0] * N + [0] * N1 + [-f_long_max] * N1 + [-f_max] * N1
    ubx = [max_v] * N + [max_v] * N + [10] * N1 + [f_long_max] * N1 + [f_max] * N1

    #make lbg, ubg
    lbg = [0] * N1 + [0] * N + [0] * 2 * N1 + [0] * N1 #+ [0] * 2 
    ubg = [0] * N1 + [0] * N + [0] * 2 * N1 + [f_max] * N1 #+ [0] * 2 

    r = S(x0=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

    x_opt = r['x']

    dx = np.array(x_opt[:N])
    dy = np.array(x_opt[N:N*2])
    dt = np.array(x_opt[2*N:N*2 + N1])
    f_long = np.array(x_opt[2*N+N1:2*N + N1*2])
    f_lat = np.array(x_opt[-N1:])

    f_t = (f_long**2 + f_lat**2)**0.5

    # print(f"Dt: {dt.T}")
    # print(f"DT0: {dt[0]}")
    t = np.cumsum(dt)
    t = np.insert(t, 0, 0)
    # print(f"Dt: {dt.T}")
    # print(f"Dx: {dx.T}")
    # print(f"Dy: {dy.T}")

    vs = (dx**2 + dy**2)**0.5

    if show:
        plt.figure(1)
        plt.title("Velocity vs dt")
        plt.plot(t, vs)
        plt.plot(t, th_i)
        plt.legend(['vs', 'ths'])
        # plt.plot(t, dx)
        # plt.plot(t, dy)
        # plt.legend(['v', 'dx', 'dy'])
        plt.plot(t, np.ones_like(t) * max_v, '--')

        plt.figure(2)
        plt.title("F_long, F_lat vs t")
        plt.plot(t[:-1], f_long)
        plt.plot(t[:-1], f_lat)
        plt.plot(t[:-1], f_t, linewidth=3)
        plt.plot(t, np.ones_like(t) * f_max, '--')
        plt.plot(t, np.ones_like(t) * -f_max, '--')
        plt.plot(t, np.ones_like(t) * f_long_max, '--')
        plt.plot(t, np.ones_like(t) * -f_long_max, '--')

        plt.legend(['Flong', "f_lat", "f_t"])

        # plt.figure(3)
        # plt.title("Theta vs t")
        # plt.plot(t, th_i)
        # plt.plot(t, np.abs(th_i))

        # plt.figure(5)
        # plt.title(f"t vs dt")
        # plt.plot(t[1:], dt)
        # plt.plot(t[1:], dt, '+')
    
    # plt.figure(9)
    # plt.clf()
    # plt.title("F_long, F_lat vs t")
    # plt.plot(t[:-1], f_long)
    # plt.plot(t[:-1], f_lat)
    # plt.plot(t[:-1], f_t, linewidth=3)
    # plt.plot(t, np.ones_like(t) * f_max, '--')
    # plt.plot(t, np.ones_like(t) * -f_max, '--')
    # plt.plot(t, np.ones_like(t) * f_long_max, '--')
    # plt.plot(t, np.ones_like(t) * -f_long_max, '--')
    # plt.legend(['Flong', "f_lat", "f_t"])


    return vs


def Max_velocity_conf(pts, conf, show=False):
    mu = conf.mu
    m = conf.m
    g = conf.g
    l_f = conf.l_f
    l_r = conf.l_r
    safety_f = 0.9
    f_max = mu * m * g * safety_f
    f_long_max = l_f / (l_r + l_f) * f_max
    max_v = conf.max_v  # parameter to be adapted so that optimiser isnt too fast
    # max_a = config['lims']['max_a']

    s_i, th_i = convert_pts_s_th(pts)
    th_i_1 = th_i[:-1]
    s_i_1 = s_i[:-1]
    N = len(s_i)
    N1 = len(s_i) - 1

    # setup possible casadi functions
    d_x = ca.MX.sym('d_x', N-1)
    d_y = ca.MX.sym('d_y', N-1)
    vel = ca.Function('vel', [d_x, d_y], [ca.sqrt(ca.power(d_x, 2) + ca.power(d_y, 2))])

    dx = ca.MX.sym('dx', N)
    dy = ca.MX.sym('dy', N)
    dt = ca.MX.sym('t', N-1)
    f_long = ca.MX.sym('f_long', N-1)
    f_lat = ca.MX.sym('f_lat', N-1)

    nlp = {\
        'x': ca.vertcat(dx, dy, dt, f_long, f_lat),
        'f': ca.sum1(dt), 
        'g': ca.vertcat(
                    # dynamic constraints
                    dt - s_i_1 / ((vel(dx[:-1], dy[:-1]) + vel(dx[1:], dy[1:])) / 2 ),
                    # ca.arctan2(dy, dx) - th_i,
                    dx/dy - ca.tan(th_i),
                    dx[1:] - (dx[:-1] + (ca.sin(th_i_1) * f_long + ca.cos(th_i_1) * f_lat) * dt  / m),
                    dy[1:] - (dy[:-1] + (ca.cos(th_i_1) * f_long - ca.sin(th_i_1) * f_lat) * dt  / m),

                    # path constraints
                    ca.sqrt(ca.power(f_long, 2) + ca.power(f_lat, 2)),

                    # boundary constraints
                    # dx[0], dy[0]
                ) \
    }

    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})
    # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})

    # make init sol
    v0 = np.ones(N) * max_v/2
    dx0 = v0 * np.sin(th_i)
    dy0 = v0 * np.cos(th_i)
    dt0 = s_i_1 / ca.sqrt(ca.power(dx0[:-1], 2) + ca.power(dy0[:-1], 2)) 
    f_long0 = np.zeros(N-1)
    ddx0 = dx0[1:] - dx0[:-1]
    ddy0 = dy0[1:] - dy0[:-1]
    a0 = (ddx0**2 + ddy0**2)**0.5 
    f_lat0 = a0 * m

    x0 = ca.vertcat(dx0, dy0, dt0, f_long0, f_lat0)

    # make lbx, ubx
    # lbx = [-max_v] * N + [-max_v] * N + [0] * N1 + [-f_long_max] * N1 + [-f_max] * N1
    lbx = [-max_v] * N + [0] * N + [0] * N1 + [-f_long_max] * N1 + [-f_max] * N1
    ubx = [max_v] * N + [max_v] * N + [10] * N1 + [f_long_max] * N1 + [f_max] * N1

    #make lbg, ubg
    lbg = [0] * N1 + [0] * N + [0] * 2 * N1 + [0] * N1 #+ [0] * 2 
    ubg = [0] * N1 + [0] * N + [0] * 2 * N1 + [f_max] * N1 #+ [0] * 2 

    r = S(x0=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

    x_opt = r['x']

    dx = np.array(x_opt[:N])
    dy = np.array(x_opt[N:N*2])
    dt = np.array(x_opt[2*N:N*2 + N1])
    f_long = np.array(x_opt[2*N+N1:2*N + N1*2])
    f_lat = np.array(x_opt[-N1:])

    f_t = (f_long**2 + f_lat**2)**0.5

    # print(f"Dt: {dt.T}")
    # print(f"DT0: {dt[0]}")
    t = np.cumsum(dt)
    t = np.insert(t, 0, 0)
    # print(f"Dt: {dt.T}")
    # print(f"Dx: {dx.T}")
    # print(f"Dy: {dy.T}")

    vs = (dx**2 + dy**2)**0.5

    if show:
        plt.figure(1)
        plt.title("Velocity vs dt")
        plt.plot(t, vs)
        plt.plot(t, th_i)
        plt.legend(['vs', 'ths'])
        # plt.plot(t, dx)
        # plt.plot(t, dy)
        # plt.legend(['v', 'dx', 'dy'])
        plt.plot(t, np.ones_like(t) * max_v, '--')

        plt.figure(2)
        plt.title("F_long, F_lat vs t")
        plt.plot(t[:-1], f_long)
        plt.plot(t[:-1], f_lat)
        plt.plot(t[:-1], f_t, linewidth=3)
        plt.plot(t, np.ones_like(t) * f_max, '--')
        plt.plot(t, np.ones_like(t) * -f_max, '--')
        plt.plot(t, np.ones_like(t) * f_long_max, '--')
        plt.plot(t, np.ones_like(t) * -f_long_max, '--')

        plt.legend(['Flong', "f_lat", "f_t"])

    return vs


def convert_pts_s_th(pts):
    N = len(pts)
    s_i = np.zeros(N-1)
    th_i = np.zeros(N-1)
    for i in range(N-1):
        s_i[i] = lib.get_distance(pts[i], pts[i+1])
        th_i[i] = lib.get_bearing(pts[i], pts[i+1])

    return s_i, th_i

