import matplotlib.pyplot as plt
import numpy as np

class Planet:

    def __init__(self, tmin, tmax, dt, mass, pos0, vel0):
        self.tmin = tmin
        self.tmax = tmax
        self.dt = dt
        self.mass = mass
        self.GM_earth = 6.67*10**(-11) * 5.972*10**(24)
        self.trans1 = False
        self.trans2 = False

        self.N = int(np.round((tmax - tmin) / dt))+1
        self.posx = np.zeros(self.N)
        self.posy = np.zeros(self.N)
        self.vx = np.zeros(self.N)
        self.vy = np.zeros(self.N)
        self.r = np.zeros(self.N)
        self.KE = np.zeros(self.N)
        self.PE = np.zeros(self.N)
        self.E_tot = np.zeros(self.N)

        self.posx[0] = pos0[0]
        self.posy[0] = pos0[1]
        self.vx[0] = vel0[0]
        self.vy[0] = vel0[1]
        self.r[0] = np.sqrt(pos0[0]**2 + pos0[1]**2)

    def func(self, q_vec):
        r = np.sqrt(q_vec[0]**2 + q_vec[1]**2)
        return np.array([q_vec[2], q_vec[3], -self.GM_earth/r**3 * q_vec[0], -self.GM_earth/r**3 * q_vec[1]])

    def runge_kutta_2dim(self, i, f):
        q_vec_n = np.array([self.posx[i], self.posy[i], self.vx[i], self.vy[i]])

        k1 = f(q_vec_n)
        k2 = f(q_vec_n + self.dt * k1 / 2)
        k3 = f(q_vec_n + self.dt * k2 / 2)
        k4 = f(q_vec_n + self.dt * k3)

        q_vec_n_plus = q_vec_n + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.posx[i+1], self.posy[i+1] = q_vec_n_plus[0], q_vec_n_plus[1]
        self.vx[i+1], self.vy[i+1] = q_vec_n_plus[2], q_vec_n_plus[3]
        self.r[i + 1] = (self.posx[i + 1] ** 2 + self.posy[i + 1] ** 2)**(1/2)

    def RK4(self):
        r_max = 0
        delta_v = 0

        for i in range(self.N-1):
            if (i*self.dt > 5452) and (not self.trans1):
                vel = (self.vx[i]**2 + self.vy[i]**2)**(1/2)
                ratio = 10133.3953/vel
                self.vx[i] = ratio * self.vx[i]
                self.vy[i] = ratio * self.vy[i]
                new_vel = (self.vx[i] ** 2 + self.vy[i] ** 2) ** (1 / 2)
                self.trans1 = True
                delta_v += new_vel - vel
                t1 = i*self.dt

            r = (self.posx[i]**2 + self.posy[i]**2)**(1/2)
            if r > r_max:
                r_max = r

            if (not self.trans2) and (r > (35680000+6371000)):
                vel = (self.vx[i]**2 + self.vy[i]**2)**(1/2)
                ratio = 3077.7593/vel
                self.vx[i] = ratio * self.vx[i]
                self.vy[i] = ratio * self.vy[i]
                self.trans2 = True
                new_vel = (self.vx[i] ** 2 + self.vy[i] ** 2) ** (1 / 2)
                delta_v += new_vel - vel
                t2 = i * self.dt

            self.runge_kutta_2dim(i,self.func)
        print(r_max)
        print(delta_v)
        print(t2-t1)

    def calculateEnergies(self):
        self.KE = 1/2*self.mass*(self.vx**2+self.vy**2)
        self.PE = -self.GM_earth*self.mass/self.r
        self.E_tot = self.KE + self.PE

    def plotEnergy(self):
        self.calculateEnergies()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        time = np.linspace(self.tmin, self.tmax, self.N)
        ax.plot(time, self.KE, color="red", label=r"$E_k(t)$")
        ax.plot(time, self.PE, color="blue", label=r"$E_p(t)$")
        ax.plot(time, self.E_tot, color="purple", label=r"$E_{tot}(t)$")
        ax.set_xlabel(r"$t$(yr)")
        ax.set_ylabel(r"$E$(M$_E$*AU$^3$/yr$^2$)")
        ax.legend()
        plt.show()

    def plotEtot(self):
        time = np.linspace(self.tmin, self.tmax, self.N)
        plt.plot(time, self.E_tot, 'k', label="Total energy (Joules)")
        plt.grid('on')
        plt.legend(fontsize=14)
        plt.show()

    def plotError(self):
        time = np.linspace(self.tmin, self.tmax, self.N)
        energy = np.abs(self.E_tot - self.E_tot[0]) / np.abs(self.E_tot[0])
        plt.plot(time, energy)
        plt.yscale('log')
        plt.show()

    def plotTrajectory(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.posx, self.posy, 'k')
        ax.set_xlabel(r"$x$ (meters)")
        ax.set_ylabel(r"$y$ (meters)")
        ax.axis('equal')
        plt.title("Trajectory of satellite around the earth")
        ax.plot([0], [0], marker='o', markersize=25, color='blue')
        plt.show()
    def plotGlobalError(self):
        time = np.linspace(self.tmin + self.dt, self.tmax, self.N - 1)
        error = 0
        globalError = []
        for i in range(1, self.N):
            error += abs(self.E_tot[i] - self.E_tot[i-1])
            globalError.append(error)
        plt.plot(time, globalError, 'k', label="Global error in total energy")
        plt.legend(fontsize=14)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Global error (Joule)")
        plt.grid('on')
        plt.show()

    def relativeEndError(self):
        return abs((self.E_tot[self.N-1] - self.E_tot[0]) / self.E_tot[0])


def plotErrors():
    dt = np.linspace(0.0001, 1, 10**4)
    error = []
    t_min = 0
    t_max = 57.523
    vel0 = 35171.82
    for t in dt:
        Satellite = Planet(tmin=t_min, tmax=t_max, dt=t, mass=1, pos0=[322000, 0], vel0=[0, vel0])
        Satellite.RK4()
        Satellite.calculateEnergies()
        error.append(Satellite.relativeEndError())
    plt.plot(dt, error, 'k', label="Error after one orbit", linewidth=0.8)
    plt.yscale('log')
    plt.grid('on')
    plt.legend(fontsize=14)
    plt.show()

if __name__ == "__main__":
    t_min = 0
    t_max = 300000
    vel0 = 7714.58
    dt = 1
    r0 = (322 + 6371) * 10 ** 3

    Satellite = Planet(tmin=t_min, tmax=t_max, dt=dt, mass=1, pos0=[r0, 0], vel0=[0, vel0])
    Satellite.RK4()
    Satellite.plotTrajectory()