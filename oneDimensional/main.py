import matplotlib.pyplot as plt
import math
import numpy as np
import random
import heapq



class Particle:
    
    def __init__(self,mass,xPos,velo,isWall=False):
        self.mass = mass
        self.xPos = xPos
        self.velo = velo
        self.isWall = isWall
        self.version = 0
    
    def set_xPos(self,xPos):
        self.xPos = xPos
    
    def set_velo(self,velo):
        self.velo = velo
    
    def timeToCollision(self,colParticle):
        dv = self.velo - colParticle.velo
        if math.isclose(dv, 0):
            return np.inf
        dx = colParticle.xPos - self.xPos
        t = dx / dv
        return t if t > 0 else np.inf

    def veloAfterCollision(self,colParticle):
        m1, m2 = self.mass, colParticle.mass
        u1, u2 = self.velo, colParticle.velo
        
        v1f = (u1 * (m1 - m2) + 2 * m2 * u2) / (m1 + m2)
        v2f = (u2 * (m2 - m1) + 2 * m1 * u1) / (m1 + m2)
        
        return [v1f, v2f]

def main():
    
    boxLeftBound = -10
    boxRightBound = 10
    maxCollisions = 200000
    
    numberOfParticles = 300
    particleList = [None] * (numberOfParticles + 2)
    particleList[0] = Particle(1,boxLeftBound,0,isWall=True)
    particleList[1] = Particle(1,boxRightBound,0,isWall=True)
    
    for i in range(2,len(particleList)):
        particleList[i] = Particle(random.uniform(0.1,10),random.uniform(-9,9),random.uniform(-1000,1000))
    
    collisionOrder = []
    for i in range(len(particleList)):
        for j in range(i):
            if (i == j) or (i == 0 and j == 1) or (j == 0 and i == 1):
                continue
            time = particleList[i].timeToCollision(particleList[j])
            if time > 0:
                heapq.heappush(collisionOrder, (time,i,j,particleList[i].version,particleList[j].version))
        
    numCollisions = 0
    while numCollisions < maxCollisions and collisionOrder:
        time, i, j, v1,v2 = heapq.heappop(collisionOrder)
        if (particleList[i].version != v1) or (particleList[j].version != v2):
            continue
        
        for particle in particleList:
            particle.set_xPos(particle.xPos + particle.velo * time)
        
        v1,v2 = particleList[i].veloAfterCollision(particleList[j])
        if not particleList[i].isWall:
            particleList[i].set_velo(v1)
            particleList[i].version += 1
        if not particleList[j].isWall:
            particleList[j].set_velo(v2)
            particleList[j].version += 1

        for k in range(len(particleList)):
            if k == i or k == j:
                continue
            for a, b in [(i, k), (j, k)]:
                time = particleList[a].timeToCollision(particleList[b])
                if time > 0 and np.isfinite(time):
                    heapq.heappush(collisionOrder, (time,a,b,particleList[a].version,particleList[b].version))
        numCollisions += 1

    absoluteList = [0.5 * p.mass * p.velo ** 2 for p in particleList]

    plt.hist(absoluteList)
    plt.show()
    
if __name__ == "__main__":
    main()
    
    