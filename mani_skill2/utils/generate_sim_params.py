import numpy as np



def generate_sim_params():
    sim_params = dict()
    sim_params['latency'] = np.random.randint(0, 10)
    #-----Low Level Control-----#
    sim_params['stiffness'] = np.random.uniform()
    sim_params['damping'] = np.random.uniform()
    sim_params['force_limit'] = np.random.uniform()
    sim_params['sim_freq'] = np.random.uniform()
    #-----Robot's physicial property-----#
    sim_params['robot_fri_static'] = np.random.uniform()
    sim_params['robot_fri_dynamic'] = np.random.uniform()
    sim_params['robot_fri_restriction'] = np.random.uniform()
    #-----Object's physicial property-----#
    sim_params['obj_fri_static'] = np.random.uniform()
    sim_params['obj_fri_dynamic'] = np.random.uniform()
    sim_params['obj_fri_restriction'] = np.random.uniform()

if __name__ == '__main__':
    generate_sim_params()