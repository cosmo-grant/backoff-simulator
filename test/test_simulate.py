# NOTE: If Python's random internals change, refresh snapshots by pasting in actual pytest output.

import random

from pytest import mark, param

from backoff_simulator.config import BackoffConfig, Spec
from backoff_simulator.simulation import Event, EventType, simulate

THROTTLING_SPEC = Spec(
    title="test_throttling",
    max_clients=3,
    repeat=2,
    network_mu=5.0,
    network_sigma=1.0,
    work_to_duration=1.0,
    control="ThrottlingServer",
    control_params={"limit": 2, "window": 5.0},
    strategies=[BackoffConfig(type="Constant", params={"constant": 1.0})],
)

LOCKING_SPEC = Spec(
    title="test_locking",
    max_clients=3,
    repeat=2,
    network_mu=5.0,
    network_sigma=1.0,
    work_to_duration=1.0,
    control="LockingServer",
    control_params={"write_mu": 1.0, "write_sigma": 0.5},
    strategies=[BackoffConfig(type="FullJitteredExpo", params={"base": 1.0, "cap": 32.0})],
)

WRITE_ONLY_OCC_SPEC = Spec(
    title="test_write_only_occ",
    max_clients=3,
    repeat=2,
    network_mu=5.0,
    network_sigma=1.0,
    work_to_duration=1.0,
    control="WriteOnlyOCCServer",
    control_params={"write_mu": 1.0, "write_sigma": 0.5},
    strategies=[BackoffConfig(type="EqualJitteredExpo", params={"base": 1.0, "cap": 32.0})],
)

READ_WRITE_OCC_SPEC = Spec(
    title="test_read_write_occ",
    max_clients=3,
    repeat=2,
    network_mu=5.0,
    network_sigma=1.0,
    work_to_duration=1.0,
    control="ReadWriteOCCServer",
    control_params={"write_mu": 1.0, "write_sigma": 0.5},
    strategies=[BackoffConfig(type="FullJitteredExpo", params={"base": 1.0, "cap": 32.0})],
)


THROTTLING_CONSTANT_HISTORY = [
    (0.0, Event(EventType.CLIENT_REQUESTS_WRITE, 0, "")),
    (0.0, Event(EventType.CLIENT_REQUESTS_WRITE, 1, "")),
    (0.0, Event(EventType.CLIENT_REQUESTS_WRITE, 2, "")),
    (3.56317055489747, Event(EventType.SERVER_ACCEPTS, 1, "count=1")),
    (3.977896829989127, Event(EventType.SERVER_ACCEPTS, 0, "count=2")),
    (5.199311976483754, Event(EventType.SERVER_REJECTS, 2, "")),
    (8.56317055489747, Event(EventType.SERVER_DECREMENTS, 1, "count=1")),
    (8.977896829989128, Event(EventType.SERVER_DECREMENTS, 0, "count=0")),
    (10.332686581142358, Event(EventType.CLIENT_BACKS_OFF, 2, "")),
    (11.332686581142358, Event(EventType.CLIENT_REQUESTS_WRITE, 2, "")),
    (16.879154881480588, Event(EventType.SERVER_ACCEPTS, 2, "count=1")),
    (21.879154881480588, Event(EventType.SERVER_DECREMENTS, 2, "count=0")),
]

LOCKING_FULL_JITTERED_HISTORY = [
    (0.0, Event(EventType.CLIENT_REQUESTS_WRITE, 0, "")),
    (0.0, Event(EventType.CLIENT_REQUESTS_WRITE, 1, "")),
    (0.0, Event(EventType.CLIENT_REQUESTS_WRITE, 2, "")),
    (4.290717851929139, Event(EventType.SERVER_ACCEPTS, 1, "")),
    (4.871797292089642, Event(EventType.SERVER_REJECTS, 0, "")),
    (4.927847804172157, Event(EventType.SERVER_COMMITS, 1, "")),
    (7.026629313810512, Event(EventType.SERVER_ACCEPTS, 2, "")),
    (8.04831473891983, Event(EventType.SERVER_COMMITS, 2, "")),
    (10.094663577458483, Event(EventType.CLIENT_BACKS_OFF, 0, "")),
    (10.63607605025198, Event(EventType.CLIENT_REQUESTS_WRITE, 0, "")),
    (16.545107076405188, Event(EventType.SERVER_ACCEPTS, 0, "")),
    (17.36233494328893, Event(EventType.SERVER_COMMITS, 0, "")),
]

WRITE_ONLY_OCC_EQUAL_JITTERED_HISTORY = [
    (0.0, Event(EventType.CLIENT_REQUESTS_WRITE, 0, "")),
    (0.0, Event(EventType.CLIENT_REQUESTS_WRITE, 1, "")),
    (0.0, Event(EventType.CLIENT_REQUESTS_WRITE, 2, "")),
    (4.274259904486037, Event(EventType.SERVER_TENTATIVELY_WRITES, 2, "")),
    (4.290717851929139, Event(EventType.SERVER_TENTATIVELY_WRITES, 0, "")),
    (5.312403277038457, Event(EventType.SERVER_COMMITS, 0, "version=1")),
    (5.385693047170458, Event(EventType.SERVER_ABORTS, 2, "")),
    (7.026629313810512, Event(EventType.SERVER_TENTATIVELY_WRITES, 1, "")),
    (7.722252696579982, Event(EventType.SERVER_COMMITS, 1, "version=2")),
    (8.099193779453826, Event(EventType.CLIENT_BACKS_OFF, 2, "")),
    (8.789795898297932, Event(EventType.CLIENT_REQUESTS_WRITE, 2, "")),
    (14.007967711658514, Event(EventType.SERVER_TENTATIVELY_WRITES, 2, "")),
    (15.520112050902023, Event(EventType.SERVER_COMMITS, 2, "version=3")),
]

READ_WRITE_OCC_FULL_JITTERED_HISTORY = [
    (0.0, Event(EventType.CLIENT_REQUESTS_VERSION, 0, "")),
    (0.0, Event(EventType.CLIENT_REQUESTS_VERSION, 1, "")),
    (0.0, Event(EventType.CLIENT_REQUESTS_VERSION, 2, "")),
    (3.9131153878255205, Event(EventType.SERVER_REPORTS_VERSION, 1, "version=0")),
    (4.598339739925981, Event(EventType.SERVER_REPORTS_VERSION, 2, "version=0")),
    (5.666776438625649, Event(EventType.SERVER_REPORTS_VERSION, 0, "version=0")),
    (8.413086818873152, Event(EventType.CLIENT_REQUESTS_WRITE, 1, "")),
    (10.573914412969565, Event(EventType.CLIENT_REQUESTS_WRITE, 0, "")),
    (11.578955459449228, Event(EventType.CLIENT_REQUESTS_WRITE, 2, "")),
    (14.065307042680725, Event(EventType.SERVER_TENTATIVELY_WRITES, 1, "")),
    (14.289888629115502, Event(EventType.SERVER_COMMITS, 1, "version=1")),
    (16.193289475143285, Event(EventType.SERVER_TENTATIVELY_WRITES, 0, "")),
    (16.298082029262336, Event(EventType.SERVER_TENTATIVELY_WRITES, 2, "")),
    (17.09448364443054, Event(EventType.SERVER_ABORTS, 2, "")),
    (17.675711460481356, Event(EventType.SERVER_ABORTS, 0, "")),
    (21.37044663537139, Event(EventType.CLIENT_BACKS_OFF, 0, "")),
    (21.673815146304307, Event(EventType.CLIENT_REQUESTS_VERSION, 0, "")),
    (22.81244030117544, Event(EventType.CLIENT_BACKS_OFF, 2, "")),
    (23.658637719603753, Event(EventType.CLIENT_REQUESTS_VERSION, 2, "")),
    (24.909938240357487, Event(EventType.SERVER_REPORTS_VERSION, 0, "version=1")),
    (27.576503369927238, Event(EventType.SERVER_REPORTS_VERSION, 2, "version=1")),
    (28.577130761268542, Event(EventType.CLIENT_REQUESTS_WRITE, 0, "")),
    (32.532238932325406, Event(EventType.CLIENT_REQUESTS_WRITE, 2, "")),
    (34.30537208458726, Event(EventType.SERVER_TENTATIVELY_WRITES, 0, "")),
    (35.45714943850076, Event(EventType.SERVER_COMMITS, 0, "version=2")),
    (37.692743611704735, Event(EventType.SERVER_TENTATIVELY_WRITES, 2, "")),
    (38.19832539921743, Event(EventType.SERVER_ABORTS, 2, "")),
    (43.785117113031234, Event(EventType.CLIENT_BACKS_OFF, 2, "")),
    (45.19119863716249, Event(EventType.CLIENT_REQUESTS_VERSION, 2, "")),
    (51.30805097829933, Event(EventType.SERVER_REPORTS_VERSION, 2, "version=2")),
    (55.86532454687261, Event(EventType.CLIENT_REQUESTS_WRITE, 2, "")),
    (60.00332080168048, Event(EventType.SERVER_TENTATIVELY_WRITES, 2, "")),
    (60.450729364103765, Event(EventType.SERVER_COMMITS, 2, "version=3")),
]


@mark.parametrize(
    "spec, strategy_name, expected_history",
    [
        param(
            THROTTLING_SPEC,
            "Constant",
            THROTTLING_CONSTANT_HISTORY,
            id="ThrottlingServer-Constant",
        ),
        param(
            LOCKING_SPEC,
            "FullJitteredExpo",
            LOCKING_FULL_JITTERED_HISTORY,
            id="LockingServer-FullJitteredExpo",
        ),
        param(
            WRITE_ONLY_OCC_SPEC,
            "EqualJitteredExpo",
            WRITE_ONLY_OCC_EQUAL_JITTERED_HISTORY,
            id="WriteOnlyOCCServer-EqualJitteredExpo",
        ),
        param(
            READ_WRITE_OCC_SPEC,
            "FullJitteredExpo",
            READ_WRITE_OCC_FULL_JITTERED_HISTORY,
            id="ReadWriteOCCServer-FullJitteredExpo",
        ),
    ],
)
def test_simulate(
    spec: Spec,
    strategy_name: str,
    expected_history: list[tuple[float, Event]],
):
    random.seed(1)
    groups = simulate(spec)
    simulations = groups[(3, strategy_name)]
    assert len(simulations) == 2
    actual_history = simulations[0].history
    assert actual_history == expected_history
