import pytest
from modules.Properties import Properties

@pytest.fixture(scope="module")
def props():
    input_dict = {
        "Thermal properties": {
            "Temperature": 300,
        },
        "System properties": {
            "Number of polymers": 1,
            "Concentration acid particles": 1e-3,
            "Concentration salt particles": 1e-3,
            "Number of monomers": 22,
            "NH monomers": 6,
            "NH2 monomers": 2,
            "bond length": 1.5,
            "bond constant": 300,
            "bending constant": 0.01, # Optional
            "dihedral constant": 100., # Optional
        },
        "Particles properties" : {
            "HA": {"index": 0, "charge": +1, "sigma": 2*2.6, "epsilon": 231},
            "A": {"index": 1, "charge": 0, "sigma": 2*3.172, "epsilon": 231},
            "B": {"index": 2, "charge": +1, "sigma": 2*2.958, "epsilon": 0.0726},
            "N": {"index": 3, "charge": 0, "sigma": 2*3.93, "epsilon": 56.0},
            "Na": {"index": 4, "charge": +1, "sigma": 2*3.9624, "epsilon": 0.738},
            "Cl": {"index": 5, "charge": -1, "sigma": 2*3.915, "epsilon": 0.305},
            "HA2": {"index": 6, "charge": +1, "sigma": 2*2.6, "epsilon": 231},
            "A2": {"index": 7, "charge": 0, "sigma": 2*3.172, "epsilon": 231},
        },

        "pH properties": {
            "pK1": 8.18,
            "pK2": 10.02,
            "NUM_PHS": 8,
            "pHmin": 2.5,
            "pHmax": 12.0,
            },
        "Montecarlo properties": {
            "N_BLOCKS": 16,
            "DESIRED_BLOCK_SIZE": 100,
            "PROB_REACTION": 0.5,
        },
        "Simulation configuration": {
            "time step": 0.001,
            "USE_WCA": False,
            "USE_ELECTROSTATICS": False,
            "USE_FENE": False,
            "USE_BENDING": False,
            "USE_DIHEDRAL_POT": False,
            "USE_P3M": False,
        },

    }
    return Properties(input_dict)

def test_properties_temperature(props):
    assert props.temperature == 300.0

    with pytest.raises(TypeError):
        props.temperature = "300.0"

    with pytest.raises(ValueError):
        props.temperature = -300.0

def test_properties_n_poly(props):
    assert props.n_poly == 10

    with pytest.raises(TypeError):
        props.n_poly = "10"

    with pytest.raises(ValueError):
        props.n_poly = -10

def test_properties_c_acid(props):
    assert props.c_acid == 0.05

    with pytest.raises(TypeError):
        props.c_acid = "0.05"

    with pytest.raises(ValueError):
        props.c_acid = -0.05

def test_properties_c_salt(props):
    assert props.c_salt == 0.1

    with pytest.raises(TypeError):
        props.c_salt = "0.1"

    with pytest.raises(ValueError):
        props.c_salt = -0.1

def test_properties_n_mon(props):
    assert props.n_mon == 20

    with pytest.raises(TypeError):
        props.n_mon = "20"

    with pytest.raises(ValueError):
        props.n_mon = -20

def test_properties_n_nh(props):
    assert props.n_nh == 3

    with pytest.raises(TypeError):
        props.n_nh = "3"

    with pytest.raises(ValueError):
        props.n_nh = -3

    with pytest.raises(ValueError):
        props.n_nh = 30

def test_properties_n_nh2(props):
    assert props.n_nh2 == 2

    with pytest.raises(TypeError):
        props.n_nh2 = "2"

    with pytest.raises(ValueError):
        props.n_nh2 = -2

    with pytest.raises(ValueError):
        props.n_nh2 = 30

def test_properties_bond_l(props):
    assert props.bond_l == 0.97

    with pytest.raises(TypeError):
        props.bond_l = "0.97 A"

    with pytest.raises(ValueError):
        props.bond_l = -0.97

def test_properties_k_bond(props):
    assert props.k_bond == 10.0

    with pytest.raises(TypeError):
        props.k_bond = "10.0 kcal/mol/A^2"

    with pytest.raises(ValueError):
        props.k_bond = -10.0
