"""Lazy DesktopEnv adapter. All generated Python goes only to the OSWorld VM."""
class OSWorldEnvironment:
    def __init__(self, config, trace):
        self.config = config
        self.trace = trace
        self.env = None

    def reset(self, task):
        from desktop_env.desktop_env import DesktopEnv

        kwargs = dict(self.config['desktop'])
        # Separate evaluator downloads and caches for each episode.
        kwargs['cache_dir'] = str(self.trace.path / 'cache')
        self.env = DesktopEnv(**kwargs)
        return self.env.reset(task_config=task)

    def capture_evidence(self, regions, step):
        width, height = self.config['desktop'].get('screen_size', [1920, 1080])
        for i, region in enumerate(regions):
            try:
                coords = [region[k] for k in ('x', 'y', 'width', 'height')]
                if any(type(v) is not int for v in coords):
                    raise ValueError('Region coordinates must be integers')
                x, y, w, h = coords
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                    raise ValueError('Evidence region outside the screen')
                image = self.env.controller.get_screenshot(region=tuple(coords))
                if not isinstance(image, bytes):
                    raise ValueError('Evidence screenshot unavailable')
                name = f'evidence-{step:03d}-{i:02d}.png'
                (self.trace.path / name).write_bytes(image)
                self.trace.event('evidence', step=step, region=coords, image=name)
            except (KeyError, ValueError) as error:
                self.trace.event('invalid_evidence', step=step, error=str(error))

    def step(self, action):
        return self.env.step(action, pause=self.config['pause_seconds'])

    def evaluate(self):
        return float(self.env.evaluate())

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
