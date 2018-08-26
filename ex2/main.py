from projecao import Projecao


if __name__ == '__main__':

	proj = Projecao()
	teste = [[1, 1.5 ,2,3 ],
			 [1, 1.5, 2,3 ],
			 [1, 1, 1,1 ]]
	result = proj.proj_perspectiva_mm(teste, 1)
	proj.printMatrix(result)
	print
	result = proj.proj_perspectiva_pixel(result, 0.01, 0.01, 0, 0)
	proj.printMatrix(result)