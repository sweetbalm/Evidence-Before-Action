"""Entry point passed to --external_plugins on rollout and trainer only."""
import os

from swift.rollout import multi_turns
from osworld_opd.scheduler import OSWorldScheduler

multi_turns['osworld_opd'] = OSWorldScheduler

if os.environ.get('OSWORLD_OPD_ROLE') == 'trainer':
    from osworld_opd.teacher import install
    install()
