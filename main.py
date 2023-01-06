from simulator import Simulator

def main():
    trial = 100
    step = 1000
    K = 4
    aleph = 0.1
    sim = Simulator(trial, step, K, aleph)
    sim.run()

if __name__ == '__main__':
    print('started run')
    main()
    print('finished run')