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

    def plotEnergy(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        time = np.linspace(self.tmin, self.tmax, self.N)
        ax.plot(time, self.KE, color="red", label=r"$E_k(t)$")
        ax.plot(time, self.PE, color="blue", label=r"$E_p(t)$")
        ax.plot(time, self.E_tot, color="purple", label=r"$E_{tot}(t)$")
        plt.xlabel("Time (yr)")
        plt.ylabel(r"Energy($Au^3/yr^2$)")
        ax.legend()
        plt.show()

    def plotPE(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        time = np.linspace(self.tmin, self.tmax, self.N)
        ax.plot(time, self.PE, color="red", label=r"$E_p(t)$")
        plt.xlabel("Time (yr)")
        plt.title("Potential energy of Earth")
        plt.ylabel(r"Energy($Au^3/yr^2$)")
        plt.grid('on')
        ax.legend()
        plt.show()

    def plotKE(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        time = np.linspace(self.tmin, self.tmax, self.N)
        ax.plot(time, self.KE, color="red", label=r"$E_k(t)$")
        plt.xlabel("Time (yr)")
        plt.title("Kinetic energy of Earth")
        plt.grid('on')
        plt.ylabel(r"Energy($Au^3/yr^2$)")
        ax.legend()
        plt.show()

    def plotEtot(self):
        time = np.linspace(self.tmin, self.tmax, self.N)
        plt.plot(time, self.E_tot, 'k', label=r"Total energy ($Au^3/yr^2$)")
        plt.title("Total energy of Earth")
        plt.xlabel("Time (yr)")
        plt.ylabel(r"Energy($Au^3/yr^2$)")
        plt.grid('on')
        plt.legend(fontsize=14)
        plt.show()

    def plotVelocities(self):
        time = np.linspace(self.tmin, self.tmax, self.N)
        vel = np.sqrt(self.vx**2 + self.vy**2)
        plt.plot(time, vel, 'k', label=r"$v_{tot}$ (Au/yr)")
        plt.title("Velocity of Earth")
        plt.ylabel("Velocity (Au/yr)")
        plt.xlabel("Time (yr)")
        plt.legend(fontsize=14)
        plt.grid('on')
        plt.show()


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

            r_sun = np.sqrt(q_vec[j]**2 + q_vec[self.planetCount+j]**2)
            velx, vely = -self.GM_s*q_vec[j]/r_sun**3, -self.GM_s*q_vec[self.planetCount+j]/r_sun**3
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
            self.planetList[i].PE = -(self.planetList[i].GM_s*self.planetList[i].mass) \
            / np.sqrt(self.planetList[i].posx**2 + self.planetList[i].posy**2)

            for j in range(0, self.planetCount):
                if j != i:
                    r_ij = np.sqrt((self.planetList[i].posx - self.planetList[j].posx)**2
                                   + (self.planetList[i].posy - self.planetList[j].posy)**2)
                    self.planetList[i].PE -= (self.planetList[i].mass * self.planetList[j].GM ) / (r_ij*2)

            self.planetList[i].E_tot = self.planetList[i].KE + self.planetList[i].PE

            self.KE += self.planetList[i].KE
            self.PE += self.planetList[i].PE
            self.E_tot += self.planetList[i].E_tot

    def plotTrajectories(self):
        colors = ["b", "r"]
        planetNames = ["Earth", "Mars"]
        for i in range(0, self.planetCount):
            plt.plot(self.planetList[i].posx, self.planetList[i].posy, colors[i], label=planetNames[i], linewidth=0.7)
        plt.xlabel(r"$x$(AU)")
        plt.ylabel(r"$y$(AU)")
        plt.title("Trajectory for earth and mars")
        plt.legend()
        plt.axis('equal')
        plt.plot([0], [0], marker='o', markersize=15, color='orange')
        plt.show()

    def plotEnergies(self):
        time = np.linspace(self.tmin, self.tmax, self.N)
        plt.plot(time, self.KE, 'r', label = r"$V_k$")
        plt.plot(time, self.PE, 'b', label = r"$V_p$")
        plt.plot(time, self.E_tot, 'k', label = r"$V_tot$")
        plt.grid('on')
        plt.title("Energies of the Earth-Mars system")
        plt.xlabel("Time (yr)")
        plt.ylabel(r"Energy($Au^3/yr^2$)")
        plt.legend()
        plt.show()

    def plotEtot(self):
        time = np.linspace(self.tmin, self.tmax, self.N)
        plt.plot(time, self.E_tot, 'k', label=r"$E_{tot}$ System")
        plt.xlabel("Time (yr)")
        plt.title("Total energy of system")
        plt.ylabel(r"Energy($Au^3/yr^2$)")
        plt.grid('on')
        plt.legend()
        plt.show()

    def plotRelativeError(self):
        time = np.linspace(self.tmin, self.tmax, self.N)
        relativeError = np.abs((self.E_tot-self.E_tot[0]) / self.E_tot[0])
        plt.plot(time, relativeError, 'k', label=r"Relative error in $E_{tot}$")
        plt.title(r"Relative error in system $E_{tot}$")
        plt.xlabel("Time")
        plt.ylabel("Relative error")
        plt.legend(fontsize=14)
        plt.yscale('log')
        plt.grid('on')
        plt.show()

    def relativeEndError(self):
        return abs((self.E_tot[self.N - 1] - self.E_tot[0]) / self.E_tot[0])

    def relativeEndRadius(self):
        return np.abs(np.sqrt(self.planetList[0].posx[0]**2 + self.planetList[0].posy[0]**2)
                    - np.sqrt(self.planetList[0].posx[self.N-1]**2 + self.planetList[0].posy[self.N-1]**2)) \
                    / np.sqrt(self.planetList[0].posx[0]**2 + self.planetList[0].posy[0]**2)


def plotErrors():
    t_min = 0
    t_max = 1.88085
    dt = np.linspace(0.00001, 0.01, 10**3)
    error = []
    for t in dt:
        print(t)
        Earth = Planet(mass=100, pos0=[1, 0], vel0=[0, 2 * np.pi], tmin=t_min, tmax=t_max, dt=t)
        Jupiter = Planet(mass=318, pos0=[5.2028, 0], vel0=[0, 2.7546], tmin=t_min, tmax=t_max, dt=t)
        Mars = Planet(mass=0.1074, pos0=[1.5237, 0], vel0=[0, (2 / np.sqrt(1.5237)) * np.pi], tmin=t_min, tmax=t_max, dt=t)
        System = SolarSystem(np.array([Earth, Mars]), tmin=t_min, tmax=t_max, dt=t)
        System.RK4()
        System.calculateEnergies()
        error.append(System.relativeEndError())
    plt.plot(dt, error, 'k', label="Error after one orbit for different timesteps", linewidth=0.3)
    plt.yscale('log')
    plt.xlabel(r"Timestep $\tau$")
    plt.ylabel(r"Relative error in total energy")
    plt.grid('on')
    plt.legend(fontsize=12)
    plt.show()


if __name__ == "__main__":
    t_min = 0
    t_max = 7.52
    dt = 0.0005

    Earth = Planet(mass=1,pos0=[1,0],vel0=[0, 2 * np.pi], tmin=t_min, tmax=t_max, dt=dt)
    Jupiter = Planet(mass=318, pos0=[5.2028, 0], vel0=[0, 2.7546], tmin=t_min, tmax=t_max, dt=dt)
    Mars = Planet(mass=0.1074, pos0=[1.5237, 0], vel0=[0, (2/np.sqrt(1.5237))*np.pi], tmin=t_min, tmax=t_max, dt=dt)
    System = SolarSystem(np.array([Earth, Mars]), tmin=t_min, tmax=t_max, dt=dt)
    System.RK4()
    System.plotTrajectories()
    System.calculateEnergies()
    Mars.plotVelocities()
    Mars.plotPE()
    Mars.plotKE()
    Mars.plotEtot()
    System.plotEnergies()
    System.plotEtot()
    System.plotRelativeError()
    plotErrors()
    print(System.relativeEndRadius())