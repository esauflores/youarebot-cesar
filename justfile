# youarebot-cesar

default:
    @just --list

import 'scripts/setup.just'
import 'scripts/docker.just'
import 'scripts/app.just'
import 'scripts/test.just'
import 'scripts/sync.just'
