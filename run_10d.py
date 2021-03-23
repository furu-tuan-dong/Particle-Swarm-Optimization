from utils.problem import Rastrigin, Rosenbrock, Eggholder, Ackley, PSO

import numpy as np
import logging
## **Implement**


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
  SEED = 18520185
  PARAM = {
      'w':  0.7298,
      'c1': 1.49618,
      'c2': 1.49618,
      'k': (1 - 0.1) * np.random.random_sample() + 0.1
  }
  # Save .gif directory
  GIF_DIR = 'gif'
  # Save .log directory
  LOG_DIR = 'logs'
  CALLBACK = {
    'Rastrigin':  lambda topology, dimension : Rastrigin(topology,dimension),
    'Rosenbrock': lambda topology, dimension : Rosenbrock(topology,dimension),
    'Eggholder':  lambda topology, dimension : Eggholder(topology,2),
    'Ackley':     lambda topology, dimension : Ackley(topology,2),
  }

  CONFIG = {
    # 'N': [128,256,512,1024,2048],
    'N': [512],
    'maxEvaluations': 10e5,
    'dimension': 10,
    'generation': 50,
    'nExperiments': 10,
    # 'funcName': ['Rastrigin','Rosenbrock'],
    'funcName': ['Rastrigin','Rosenbrock'],
    'topology': ['star','ring'],
    'saveLogPath': lambda funcName,N,topology: f'{LOG_DIR}/{funcName}/{funcName.lower()}_{N}pop_{topology}_10D.log'
  }
  for N in CONFIG['N']:
    for topology in CONFIG['topology']:
        for funcName in CONFIG['funcName']:
            # try:
                # if ((funcName,topology,N) in [('Rastrigin','star',128)]): continue
                with open(CONFIG['saveLogPath'](funcName,N,topology), 'w') as f:
                    funcConfig = {
                        'funcName': funcName,
                        'topology': topology,
                        'N': N,
                        'maxEvaluations': CONFIG['maxEvaluations'],
                        'dimension': CONFIG['dimension'],
                        # 'generation': CONFIG['generation'],
                        'nExperiments': CONFIG['nExperiments'],
                    }
                    f.write((f"[Experiment] CONFIG: {funcConfig}\n"))
                for i in range(CONFIG['nExperiments']):
                    np.random.seed(SEED + i)
                    func = CALLBACK[funcName](topology,CONFIG['dimension'])
                    pso = PSO(func,paramConfig=PARAM,maxEvaluations=CONFIG['maxEvaluations'],generation=CONFIG['generation'],N=N)
                    pso.runAll()
                    reportContent = f"Run #{str(i + 1).zfill(2)}: {pso.getReport(SEED + i)}\n"
                    with open(CONFIG['saveLogPath'](funcName,N,topology), "a") as f:
                        f.write(reportContent)
                    logger.info(f'{funcName}-{N}-{topology}\n{reportContent}')
            # except:
            #     print(f'Error at {funcName}, {topology} topology, {N} population')