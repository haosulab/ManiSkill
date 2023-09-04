import numpy as np

SIM_PARAMS_NAME = [
    #-----Latency-----#
    'obs_latency',  # in seconds
    #-----Low Level Control-----#
    'stiffness',
    'damping',
    'force_limit',
    'sim_freq',
    #-----Robot's physicial property-----#
    'robot_fri_static',
    'robot_fri_dynamic',
    'robot_restitution',
    #-----Object's physicial property(defualt)-----#
    'obj_fri_static',
    'obj_fri_dynamic',
    'obj_restitution',
    #-----Low level control-----#
    'time_out',
]
SIM_PARAMS = [0.0, 1e6, 2e3, 1e2, 200, 2.0, 2.0, 0.0, 1.0, 1.0, 0.0, 50]

def generate_sim_params(params_group=None):
    if params_group == None:
        params = SIM_PARAMS
    assert len(SIM_PARAMS_NAME) == len(params)
    sim_params = dict()
    for i in range(len(SIM_PARAMS_NAME)):
        sim_params[SIM_PARAMS_NAME[i]] = params[i]
    # TODO: need to tackle with the residue of float
    # assert (sim_params['obs_latency'] % (1 / sim_params['sim_freq'])) < 1e-7
    return sim_params

if __name__ == '__main__':
    print(generate_sim_params())