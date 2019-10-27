import matplotlib.pyplot as plt
import numpy as np

class Planet:

    def __init__(self, mass, pos0, vel0, tmin, tmax, dt):
        self.tmin = tmin
        self.tmax = tmax
        self.N = int(np.round((tmax - tmin) / dt)) + 1
        self.dt = (tmax - tmin) / (self.N - 1)
        self.mass = mass
        self.GM_s = 4*np.pi**2
        self.GM = self.GM_s * self.mass / 332946
        self.posx = np.zeros(self.N)
        self.posy = np.zeros(self.N)
        self.vx = np.zeros(self.N)
        self.vy = np.zeros(self.N)
        self.KE = np.zeros(self.N)
        self.PE = np.zeros(self.N)
        self.E_tot = np.zeros(self.N)

        self.posx[0] = pos0[0]
        self.posy[0] = pos0[1]
        self.vx[0] = vel0[0]
        self.vy[0] = vel0[1]


class SolarSystem:

    def __init__(self, planetList, tmin, tmax, dt):
        self.tmin = tmin
        self.tmax = tmax
        self.N = int(np.round((tmax - tmin) / dt)) + 1
        # adjusting dt to make sure that it covers the entire interval
        self.dt = (tmax - tmin) / (self.N - 1)
        self.planetList = planetList
        self.planetCount = len(planetList)
        self.GM_s = 4*np.pi**2
        self.KE = np.zeros(self.N)
        self.PE = np.zeros(self.N)
        self.E_tot = np.zeros(self.N)

    def func(self, q_vec):
        next = np.zeros(4 * self.planetCount)

        for j in range(0, self.planetCount):
            next[j] = q_vec[2*self.planetCount+j]
            next[self.planetCount+j] = q_vec[3*self.planetCount+j]

            # r_sun = np.sqrt(q_vec[j]**2 + q_vec[self.planetCount+j]**2)
            # velx, vely = -self.GM_s*q_vec[j]/r_sun**3, -self.GM_s*q_vec[self.planetCount+j]/r_sun**3
            velx, vely = 0, 0
            for h in range(0, self.planetCount):
                if h != j:
                    r_jh = np.sqrt((q_vec[h] - q_vec[j])**2 + (q_vec[self.planetCount+h] - q_vec[self.planetCount+j])**2)
                    velx -= self.planetList[h].GM*(q_vec[j] - q_vec[h])/r_jh**3
                    vely -= self.planetList[h].GM * (q_vec[self.planetCount+j] - q_vec[self.planetCount+h])/ r_jh ** 3
            next[2*self.planetCount+j] = velx
            next[3*self.planetCount+j] = vely

        return next

    def runge_kutta(self, i, f):

        q_vec_n = np.array([planet.posx[i] for planet in self.planetList]
                           + [planet.posy[i] for planet in self.planetList]
                           + [planet.vx[i] for planet in self.planetList]
                           + [planet.vy[i] for planet in self.planetList])

        k1 = f(q_vec_n)
        k2 = f(q_vec_n + self.dt * k1 / 2)
        k3 = f(q_vec_n + self.dt * k2 / 2)
        k4 = f(q_vec_n + self.dt * k3)

        q_vec_n_plus = q_vec_n + (self.dt/6) * (k1 + 2 * k2 + 2 * k3 + k4)

        for j in range(0, self.planetCount):
            self.planetList[j].posx[i + 1] = q_vec_n_plus[j]
            self.planetList[j].posy[i + 1] = q_vec_n_plus[self.planetCount+j]
            self.planetList[j].vx[i + 1] = q_vec_n_plus[2*self.planetCount+j]
            self.planetList[j].vy[i + 1] = q_vec_n_plus[3*self.planetCount+j]

    def RK4(self):
        for i in range(0, self.N - 1):
            self.runge_kutta(i, self.func)

    def calculateEnergies(self):
        for i in range(0, self.planetCount):
            self.planetList[i].KE = (1/2)*self.planetList[i].mass*(self.planetList[i].vx**2+self.planetList[i].vy**2)
            # self.planetList[i].PE = -(self.planetList[i].GM_s*self.planetList[i].mass) \
            # / np.sqrt(self.planetList[i].posx**2 + self.planetList[i].posy**2)
            self.planetList[i].PE = 0

            for j in range(0, self.planetCount):
                if j != i:
                    r_ij = np.sqrt((self.planetList[i].posx - self.planetList[j].posx)**2
                                   + (self.planetList[i].posy - self.planetList[j].posy)**2)
                    self.planetList[i].PE -= (self.planetList[i].mass * self.planetList[j].GM ) / r_ij

            self.planetList[i].E_tot = self.planetList[i].KE + self.planetList[i].PE

            self.KE += self.planetList[i].KE
            self.PE += self.planetList[i].PE
            self.E_tot += self.planetList[i].E_tot

    def plotTrajectories(self):
        colors = ["r", "b", 'k']
        planetNames = ["Earth", "Jupiter", "Sun"]
        for i in range(0, self.planetCount):
            plt.plot(self.planetList[i].posx, self.planetList[i].posy, colors[i], label=planetNames[i])
        plt.xlabel(r"$x$(AU)")
        plt.ylabel(r"$y$(AU)")
        plt.axis('equal')
        plt.legend()
        plt.title("Trajectories of Earth, Jupiter and the Sun")
        # plt.plot([0], [0], marker='o', markersize=15, color='orange')
        plt.show()

    def plotEnergies(self):
        time = np.linspace(self.tmin, self.tmax, self.N)
        plt.plot(time, self.KE, 'r')
        plt.plot(time, self.PE, 'b')
        plt.plot(time, self.E_tot, 'k')
        plt.show()

    def plotEtot(self):
        time = np.linspace(self.tmin, self.tmax, self.N)
        plt.plot(time, self.E_tot, 'k')
        plt.show()

    def plotRelativeError(self):
        time = np.linspace(self.tmin, self.tmax, self.N)
        relativeError = np.abs((self.E_tot-self.E_tot[0]) / self.E_tot[0])
        plt.plot(time, relativeError, 'k', label="Relative error in total energy")
        plt.xlabel("Time")
        plt.ylabel("Relative error")
        plt.legend(fontsize=14)
        plt.yscale('log')
        plt.grid('on')
        plt.show()

    def relativeEndError(self):
        return abs((self.E_tot[self.N - 1] - self.E_tot[0]) / self.E_tot[0])

def plotErrors():
    t_min = 0
    t_max = 5
    dt = np.linspace(0.001, 0.1, 10**2)
    error = []
    for t in dt:
        print(t)
        Earth = Planet(mass=100, pos0=[1, 0], vel0=[0, 2 * np.pi], tmin=t_min, tmax=t_max, dt=t)
        Jupiter = Planet(mass=318, pos0=[5.2028, 0], vel0=[0, 2.7546], tmin=t_min, tmax=t_max, dt=t)
        System = SolarSystem(np.array([Earth, Jupiter]), tmin=t_min, tmax=t_max, dt=t)
        System.RK4()
        System.calculateEnergies()
        error.append(System.relativeEndError())
    plt.plot(dt, error, 'k', label="Error after one orbit for different timesteps", linewidth=0.8)
    plt.yscale('log')
    plt.xlabel(r"Timestep $\tau$")
    plt.ylabel(r"Relative error in total energy")
    plt.grid('on')
    plt.legend(fontsize=14)
    plt.show()


if __name__ == "__main__":
    t_min = 0
    t_max = 15
    dt = 0.002

    Earth = Planet(mass=1,pos0=[1,0],vel0=[0, 2 * np.pi], tmin=t_min, tmax=t_max, dt=dt)
    Jupiter = Planet(mass=318, pos0=[5.2028, 0], vel0=[0, 2.7546], tmin=t_min, tmax=t_max, dt=dt)
    Sun = Planet(mass=333500, pos0=[0, 0], vel0=[0, 0], tmin=t_min, tmax=t_max, dt=dt)
    System = SolarSystem(np.array([Earth, Jupiter, Sun]), tmin=t_min, tmax=t_max, dt=dt)
    System.RK4()
    System.calculateEnergies()
    System.plotTrajectories()
    # System.plotEnergies()
    # System.plotEtot()
    # System.plotRelativeError()