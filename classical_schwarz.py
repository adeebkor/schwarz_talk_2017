from bqplot import *
import ipywidgets as widgets

def main(overlap):
    import mpi4py.MPI as mpi
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as linsolve
    size = mpi.COMM_WORLD.size
    rank = mpi.COMM_WORLD.rank

    N = 100
    L = 1.
    dx = 1./(N - 1)
    Nl = (N-1)//size + 1
    if mpi.COMM_WORLD.rank == size-1:
        Nl += N%size
    
    Narray = mpi.COMM_WORLD.allgather(Nl-1)
    tmp = np.cumsum(Narray)
    tmp = np.insert(tmp, 0, 0)

    beg = tmp[rank]*dx
    end = tmp[rank+1]*dx


    if rank > 0:
        beg -= dx*(overlap)
    if rank < size - 1:
        end += dx*(overlap)

    xl = np.arange(beg, end + .5*dx, dx)
    
    n = xl.size

    A = sp.spdiags([-np.ones(n), 2*np.ones(n), -np.ones(n)],[-1, 0, 1], n, n) # 1D poisson matrix
    A = A.tocsr()

    A[0 ,  0] = 1; A[ 0,  1] = 0;
    A[-1, -2] = 0; A[-1, -1] = 1;

    b = np.ones(n)*dx**2
    b[0] = 0; b[-1] = 0
    
    LU = linsolve.factorized(A.tocsc())
    u = LU(b)
    
    U = []
    U.append(mpi.COMM_WORLD.allgather(u))
    X = mpi.COMM_WORLD.allgather(xl)
    nbite = 100
    for k in range(nbite):
        if rank == 0:
            mpi.COMM_WORLD.send(u[-1-2*overlap], rank + 1, rank)
            b[-1] = mpi.COMM_WORLD.recv(None, rank + 1, rank + 1)
        elif rank == size - 1:
            mpi.COMM_WORLD.send(u[2*overlap], rank - 1, rank)
            b[0] = mpi.COMM_WORLD.recv(None, rank - 1, rank - 1)
        else:
            mpi.COMM_WORLD.send(u[-1-2*overlap], rank + 1, rank)
            b[-1] = mpi.COMM_WORLD.recv(None, rank + 1, rank + 1)
            mpi.COMM_WORLD.send(u[2*overlap], rank - 1, rank)
            b[0] = mpi.COMM_WORLD.recv(None, rank - 1, rank - 1)
        u[:] = LU(b)
        U.append(mpi.COMM_WORLD.allgather(u))
    return X, U

def plot_solution(view):
    line_para = []
    size = view['size'][0]
    xx, sol = view.apply(main, 1)[0]

    x_sc, y_sc = LinearScale(), LinearScale()

    ax_x = Axis(label='x', scale=x_sc, grid_lines='solid')
    ax_y = Axis(label='solution', scale=y_sc, orientation='vertical', grid_lines='solid')
    for i in range(size):
        line_para.append(Lines(x=xx[i], y=sol[0][i], scales={'x': x_sc, 'y': y_sc}))

    play = widgets.Play(
        value=0,
        min=0,
        max=100,
        step=1,
        description="Press play",
        disabled=False,
        continuous_update=False
    )
    iteration = widgets.IntSlider(continuous_update=False)
    widgets.jslink((play, 'value'), (iteration, 'value'))

    def update_ite(iteration):
        import time
        for i in range(size):
            line_para[i].y=sol[iteration][i]
        time.sleep(.1)

    def change_overlap(overlap):
        xx[:], sol[:] = view.apply(main, overlap)[0]
        iteration.value = 0
        update_ite(0)
        for i in range(size):
            line_para[i].x=xx[i]
            line_para[i].y=sol[0][i]

    widgets.interact(update_ite, iteration=iteration);
    overlap = widgets.IntSlider(value=1, min=1, max=20, step=1, continuous_update=False)
    widgets.interact(change_overlap, overlap=overlap)

    fig_para = Figure(axes=[ax_x, ax_y], marks=line_para, animation_duration=100)
    return widgets.VBox([widgets.HBox([play, iteration, overlap]), fig_para])
