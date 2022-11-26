import unittest

import jax.numpy as jnp

from fortuna.calibration.state import CalibState
from fortuna.output_calibrator.output_calib_manager.state import \
    OutputCalibManagerState
from fortuna.prob_model.joint.state import JointState


class TestStates(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_joint_state(self):
        d = dict(
            model=dict(params=jnp.array([0.0]), batch_stats=jnp.array([0.0])),
            lik_log_var=dict(params=jnp.array([1.0]), batch_stats=jnp.array([1.0])),
        )
        js = JointState.init_from_dict(d)
        assert js.params == dict(
            model=dict(params=jnp.array([0.0])),
            lik_log_var=dict(params=jnp.array([1.0])),
        )
        assert js.mutable == dict(
            model=dict(batch_stats=jnp.array([0.0])),
            lik_log_var=dict(batch_stats=jnp.array([1.0])),
        )

    def test_output_calib_manager_state(self):
        cs = OutputCalibManagerState.init_from_dict(
            dict(
                output_calibrator=dict(
                    params=jnp.array([0.0]), batch_stats=jnp.array([0.0])
                )
            )
        )
        assert cs.params == dict(output_calibrator=dict(params=jnp.array([0.0])))
        assert cs.mutable == dict(output_calibrator=dict(batch_stats=jnp.array([0.0])))

    def test_calib_state(self):
        cs = CalibState.init_from_dict(dict(params=dict(a=1), mutable=dict(b=2)))
        assert hasattr(cs.params, "unfreeze")
        assert "a" in cs.params
        assert hasattr(cs.mutable, "unfreeze")
        assert "b" in cs.mutable
