from projecao import Projecao


if __name__ == '__main__':

	proj = Projecao()

	pjp = proj.pontos_calibracao(3,3,3)

	proj.printMatrix(pjp)