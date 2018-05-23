import time
import random
import math
import matplotlib.pyplot as plt

'''Los origenes son fijos, el dataset contiene viajes de ('dest','LGA')
y de ('LGA''dest') donde dest = {BOS,DAL,CAK,MIA,ORD,OMA}'''
people = [('Adolfo','BOS'),
          ('Pringle','DAL'),
          ('Erasmo','CAK'),
          ('Nacho','MIA'),
          ('Alan','ORD'),
          ('Everardo','OMA')]

'''Cambiar por domain = (0,9)'''
domain=[(0,9)]*(len(people)*2)
# Laguardia
destination='LGA'

flights={}
# 
file = open('schedule.txt')
for line in file:
  origin,dest,depart,arrive,price=line.strip().split(',')
  flights.setdefault((origin,dest),[])

  # Add details to the list of possible flights
  flights[(origin,dest)].append((depart,arrive,int(price)))

'''Regresa un numero entero,para utilizarlo en la funcion objetivo'''
def getminutes(t):
    x=time.strptime(t,'%H:%M')
    return x[3]*60+x[4]

def printschedule(r):
  for d in range(0,len(r),2):
    index = int(d/2)
    name=people[index][0]
    origin=people[index][1]
    out=flights[(origin,destination)][int(r[d])]
    ret=flights[(destination,origin)][int(r[d+1])]
    print ('%10s%10s %5s-%5s $%3s %5s-%5s $%3s' % (name,origin,
                                                  out[0],out[1],out[2],
                                                  ret[0],ret[1],ret[2]))
def schedulecost(sol):
  totalprice=0
  latestarrival=0
  earliestdep=24*60

  for d in range(0,len(sol),2):
    '''Obtener los vuelos de ida y regreso'''
    index = int(d/2)
    origin=people[index][1]
    outbound=flights[(origin,destination)][int(sol[d])]
    returnf=flights[(destination,origin)][int(sol[d+1])]
    
    '''Considerar el precio de los vuelos como parte de la función objetivo'''
    totalprice+=outbound[2]
    totalprice+=returnf[2]
    
    '''Considera la llegada mas tarde y la salida mas temprano'''
    if latestarrival<getminutes(outbound[1]): latestarrival=getminutes(outbound[1])
    if earliestdep>getminutes(returnf[0]): earliestdep=getminutes(returnf[0])
  
  '''Cada persona debe esperar en el aeropuerto hasta que la ultima persona llegue
  también deben llegar al mismo tiempo y esperar por sus vuelos'''  
  
  totalwait=0  
  for d in range(0,len(sol),2):
    index = int(d/2)
    origin=people[index][1]
    outbound=flights[(origin,destination)][int(sol[d])]
    returnf=flights[(destination,origin)][int(sol[d+1])]
    totalwait+=latestarrival-getminutes(outbound[1])
    totalwait+=getminutes(returnf[0])-earliestdep  

  '''Esta solucion requiere un día más? entonces renta un taxi'''
  if latestarrival>earliestdep: totalprice+=50
  
  return totalprice+totalwait

def geneticoptimize(domain,costf,fileref,popsize=300,step=1,
                    mutprob=0.2,elite=0.2,maxiter=200):
  plt_scores = []

  def mutate(vec):
    
    vec2 = vec[:]
    for i in range(len(vec2)):
        if random.random() < 0.05:
            if random.random() < 0.5:
                if vec2[i]-step>=domain[i][0]:
                    vec2[i] = vec2[i]-step
            else:
                if vec2[i]+step <= domain[i][1]:
                    vec2[i] = vec2[i]+step
    return vec2    
  
  def crossover(r1,r2):
    i=random.randint(1,len(domain)-2)
    return r1[0:i]+r2[i:]

  '''Poblacion inicial'''
  pop=[]
  for i in range(popsize):
    vec=[random.randint(domain[i][0],domain[i][1]) 
         for i in range(len(domain))]
    pop.append(vec)
  
  '''Cuantos ganadores por cada iteración'''
  topelite=int(elite*popsize)
  
  for i in range(maxiter):
    scores = []
    for v in pop:
        scores.append((costf(v),v))

    scores.sort()
    ranked=[v for (s,v) in scores]
    
    pop=ranked[0:topelite]
    
    '''Selección de individuos para la siguinte generación'''
    while len(pop)<popsize:
      c=random.randint(0,topelite)
      new = mutate(ranked[c])
      if not new is None:
          pop.append(new)
      '''En el 5% de las ocasiones, los débiles pueden ser seleccionados'''
      if random.random() < 0.05:
          c1 = random.randint(0,popsize-1)
          c2 = random.randint(0,popsize-1)
      else:
          c1=random.randint(0,topelite)
          c2=random.randint(0,topelite)
      new = crossover(ranked[c1],ranked[c2])
      pop.append(new)
    
    '''Imprime el mejor'''
    plt_scores.append(scores[0][0])
    string = str(i)+'\t'+str(scores[0][0])+'\n'
    fileref.write(string)
#    print (scores[0][0])
#  plt.plot(range(maxiter),plt_scores)
#  plt.ylabel("Mejor fitness por iteración")
#  plt.xlabel("numero de iteraciones")
  return scores[0][0]

'''Para el pso'''
class Particle:            
    def __init__(self):
        self.p =[random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
        self.vel = [random.randint(-1,1) for i in range(len(domain))]
        self.pbest =[random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
'''Funciones auxiliares para el kbpso'''
def convertToBinary(v):
    final = []
    for i in range(len(v)):
        aux = []
        a = v[i]
        for j in range(4):
            if a>0:
                aux.append(a&1)
                a = a>>1
            else: 
                aux.append(0)
        final = final + list(reversed(aux))
    return final 
def binaryToInteger(v):
    res = []
    for i in range(0,len(v),4):
        acc = 8*v[i] + 4*v[i+1] + 2*v[i+2] +v[i+3]
        res.append(acc)
    return res
'''Para el kbpso'''
class BinaryParticle:             
    def __init__(self):
        pd =[random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
        self.p = convertToBinary(pd)
        self.vel = [0]*len(self.p)
        pdbest =[random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
        self.pbest = convertToBinary(pdbest)

def psooptimize(domain,costf,popsize=300,maxiter=100):
    pop = [Particle() for i in range(popsize)]
    plt_score = []
    gbestscore = 1000000
    gbest = []
    for i in range(maxiter):
        for p_ in pop:
            distP = costf(p_.p)
            distpBest = costf(p_.pbest)
            if distP > distpBest:
                p_.pbest = p_.p
            if distP < gbestscore:
                gbestscore = distP
                gbest = p_.p
        for p_ in pop:
            c1 = random.random()
            c2 = random.random()
            for j in range(len(p_.vel)):
                p_.vel[j] =  p_.vel[j] + c1*(p_.pbest[j]-p_.p[j]) + c2*(gbest[j]-p_.p[j])
                if p_.vel[j] < 0:
                    p_.vel[j] = -1
                elif p_.vel[j] > 0:
                    p_.vel[j] = 1
            for j in range(len(p_.vel)):
                if p_.p[j]+p_.vel[j] >= domain[j][0] and p_.p[j]+p_.vel[j]<=domain[j][1]:
                    p_.p[j] = p_.p[j]+p_.vel[j]
        plt_score.append(gbestscore)
        print(gbestscore)
    plt.plot(range(maxiter),plt_score)
    return gbest
def sigmoid(b):
    return 1/(1+math.exp(-b))
'''Se utiliza KBPSO con selección de parámetros'''
def kpsooptimize(domain,costf,p1,p2,fileref,popsize=300,maxiter=200):
    pop = [BinaryParticle() for i in range(popsize)]
    plt_score = []
#    phi1 = 0.3
#    phi2 = 2
    phi1 = p1
    phi2 = p2
    gbestscore = 1000000
    gbest = []
    for i in range(maxiter):
        for p_ in pop:
            auxp = binaryToInteger(p_.p)
            auxpbest = binaryToInteger(p_.pbest)
            distP = costf(auxp)
            distpBest = costf(auxpbest)
            if distP < distpBest:
                p_.pbest = p_.p[:]
            if distP < gbestscore:
                gbestscore = distP
                gbest = p_.p[:]
        for p_ in pop:
            for j in range(len(p_.vel)):
                c1 = random.uniform(0,phi1)
                c2 = random.uniform(0,phi2)
                p_.vel[j] =  p_.vel[j] + c1*(p_.pbest[j]-p_.p[j]) + c2*(gbest[j]-p_.p[j])
                if random.uniform(0,1) < sigmoid(p_.vel[j]):
                    p_.p[j] = 1
                else: 
                    p_.p[j] = 0
            auxpp = binaryToInteger(p_.p)
            for j in range(len(auxpp)):
                if auxpp[j] > 9:
                    auxpp[j] = random.randint(domain[j][0],domain[j][1])
            p_.p = convertToBinary(auxpp)
        plt_score.append(gbestscore)
        string = str(i)+'\t'+str(gbestscore)+'\n'
        fileref.write(string)
#        print(gbestscore)
#    plt.plot(range(maxiter),plt_score)
#    plt.ylabel("Mejor fitness por iteración")
#    plt.xlabel("numero de iteraciones")
    return gbestscore

'''Para procesar con KBPSO'''
import numpy as np
MAX_RUNS = 30
phi1 = [0,0.2,1]
phi2 = [0,0.2,1]
fitness_medio_combinacion_parametros = []
fitness_std_combinacion_parametros = []
tiempo_medio_combinacion_parametros = []
tiempo_std_combinacion_parametros = []
for p1 in phi1:
    for p2 in phi2:
        fitness_acumulado =  []
        tiempos_acumulado = []
        file_tiempos = open('../resultados/KBPSO/'+str(p1)+'_'+str(p2)+'/tiempos_ejecucion.txt','w')
        file_tiempos.write('Iteración \t tiempo de ejecución \n')
        for i in range(MAX_RUNS):    
            filename = '../resultados/KBPSO/'+str(p1)+'_'+str(p2)+'/'+str(i+1)+'.txt'
            file = open(filename,'w')
            file.write('Iteracion \t mejor fitness encontrado \n')
            start_time = time.time()
            fitness = kpsooptimize(domain,schedulecost,p1,p2,file)
            elapsed  = time.time() - start_time
            string = str(i)+ '\t'+str(elapsed)+'\n'
            file_tiempos.write(string)
            tiempos_acumulado.append(elapsed)
            fitness_acumulado.append(fitness)
            file.close()
            print("Iteración p1=%f,p2=%f,i=%d \n"%(p1,p2,i))
        file_tiempos.close()
        plt.xlabel('Número de ejecución')
        plt.legend()
        plt.ylabel('Mejor fitness encontrado por ejecución')
        plt.bar(range(MAX_RUNS),fitness_acumulado,color='red')
        plt.plot()
        plt.savefig('../resultados/KBPSO/'+str(p1)+'_'+str(p2)+'/'+'grafica_fitness.png')
        print("Tiempos de p1=%f,p2=%f :\n"%(p1,p2))
        print(tiempos_acumulado)
        print("-"*10)
        fitness_promedio = np.mean(fitness_acumulado)
        desviacion_estandar = np.std(fitness_acumulado)
        tiempo_promedio = np.mean(tiempos_acumulado)
        desviacion_tiempo = np.std(tiempos_acumulado)
        fitness_medio_combinacion_parametros.append(fitness_promedio)
        fitness_std_combinacion_parametros.append(desviacion_estandar)
        tiempo_medio_combinacion_parametros.append(tiempo_promedio)
        tiempo_std_combinacion_parametros.append(desviacion_tiempo)

print("-"*10)
print("-"*10)
print("Combinaciones de parametros")
print("Fitness")
print("Fitness medio: ")
#De ahi obtener gráfica de barras para cada combinación
print(fitness_medio_combinacion_parametros)
print("Fitness desviaciones estantar")
print(fitness_std_combinacion_parametros)
print("-"*10)
print("Tiempo de ejecucion")
print("Tiempo medio: ")
print(tiempo_medio_combinacion_parametros)
print("Tiempo desviaciones estantar")
print(tiempo_std_combinacion_parametros)


'''Para procesar con GA'''
##Gráficas del fitness
#plt.xlabel('Número de ejecución')
#plt.legend()
#plt.ylabel('Mejor fitness encontrado por ejecución')
#plt.bar(range(MAX_RUNS),fitness_acumulado,color='red')
#plt.plot()
#plt.savefig("../resultados/KBPSO/grafica_fitness.png")
#
#fitness_promedio = np.mean(fitness_acumulado)
#desviacion_estandar = np.std(fitness_acumulado)
#tiempo_promedio = np.mean(tiempos_acumulado)
#desviacion_tiempo = np.std(tiempos_acumulado)
#
#print("Fitness: \n")
#print("Fitness promedio %f : \n Desviación estandar: %f"%(fitness_promedio,desviacion_estandar))
#print("-"*10)
#print("Tiempo de ejecución promedio: \n")
#print("Tiempo promedio %f : \n Desviación estándar: %f"%(tiempo_promedio,desviacion_tiempo))
#
#print("-"*10)
#print("Para graficar")
#file = open('../resultados/KBPSO/arreglo_tiempos_ejecucion.txt','w')
#file.write("[")
#for t in tiempos_acumulado:
#    string = str(t)+','
#    file.write(string)
#file.write("]")
#file.close()
#print(tiempos_acumulado)
