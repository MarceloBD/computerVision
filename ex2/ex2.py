from copy import copy
import numpy as np

# @P matriz 3xN em [mm]
# @f distancia focal da camera 
def proj_perspectiva_mm(P, f):
	p = copy(P)
	for i in xrange(len(P[0])):
		p[0][i] = f*P[0][i]/P[2][i]
		p[1][i] = f*P[1][i]/P[2][i]
		p[2][i] = f
	return p


def printMatrix(matrix):
	for i in xrange(len(matrix)):
		for j in xrange(len(matrix[0])):
			print matrix[i][j],
		print			
# @p coordenadas a projecao do pontos
# @sx dimensao horizontal do pixel [mm/pixel]
# @sy dimensao vertical do pixel [mm/pixel]
# @ox coordenada de projecao do eixo optico [pixel]
def proj_perspectiva_pixel(p, sx, sy, ox, oy):
	pixel = np.zeros(shape=(2,len(p[0])))
	for i in xrange(len(p[0])):
		pixel[0][i] = -(p[0][i]-ox)*sx 
		pixel[1][i] = -(p[1][i]-oy)*sy
	return pixel

teste = [[1, 2,3 ],
		 [1, 2,3 ],
		 [1, 1,1 ]]
result = proj_perspectiva_mm(teste, 1)
printMatrix(result)
print
result = proj_perspectiva_pixel(result, 2, 2, 0, 0)
printMatrix(result)