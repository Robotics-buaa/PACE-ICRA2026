"""Batched real-contact state machine; numpy backend enables simulator-free tests."""


class PhysicalEvents:
    def __init__(self, count, xp, make, table):
        self.xp, self.make, self.table = xp, make, table
        self.flags = {key: make([False]*count).bool() if xp.__name__ == 'torch'
                      else make([False]*count).astype(bool) for key in
                      ('own', 'hit', 'valid', 'outgoing', 'landed', 'success',
                       'paddle_touch', 'table_touch', 'own_return')}
        self.begin_step()

    def begin_step(self):
        base = self.flags['hit']
        self.pulses = {key: base & False for key in
                       ('raw_hit', 'hit', 'return', 'invalid', 'double')}

    def reset(self, ids):
        for data in (self.flags, self.pulses):
            for tensor in data.values():
                tensor[ids] = False

    def region(self, side, position):
        out = self.flags['hit'] | True
        for j, axis in enumerate('xyz'):
            lo, hi = getattr(self.table, 'table_'+side+'_contact_'+axis)
            out &= (position[:, j] >= lo) & (position[:, j] <= hi)
        return out

    def update(self, pos, previous_v, velocity, force, table_z, unsafe):
        f, p = self.flags, self.pulses
        paddle = force[:, 0] > .01
        table = (force[:, 1] > .01) & (table_z >= .8*force[:, 1]) & (previous_v[:, 2] < -.01)
        new_paddle = paddle & ~f['paddle_touch'] & ~unsafe
        new_table = table & ~f['table_touch'] & ~unsafe
        had_hit = f['hit'].clone() if hasattr(f['hit'], 'clone') else f['hit'].copy()
        # A same-substep table/paddle event is ambiguous; require prior own bounce.
        valid = new_paddle & ~had_hit & f['own'] & (previous_v[:, 0] < -.1)
        double = new_paddle & had_hit
        first = new_paddle & ~had_hit
        f['own'] |= new_table & self.region('own', pos) & ~had_hit
        f['hit'] |= new_paddle
        f['valid'] |= valid
        f['valid'] &= ~double
        f['outgoing'] |= f['valid'] & (velocity[:, 0] > .2)
        landing = new_table & had_hit & ~f['landed']
        success = landing & self.region('opponent', pos) & f['valid'] & f['outgoing'] & (velocity[:, 0] > .2)
        f['landed'] |= landing
        f['success'] |= success
        f['own_return'] |= landing & self.region('own', pos)
        f['success'] &= ~unsafe & ~double
        p['raw_hit'] |= first
        p['hit'] |= valid
        p['return'] |= success
        p['invalid'] |= first & ~valid
        p['double'] |= double
        f['paddle_touch'][:] = paddle
        f['table_touch'][:] = table

    def suppress_unsafe(self, unsafe):
        self.flags['valid'] &= ~unsafe
        self.flags['success'] &= ~unsafe
        self.pulses['hit'] &= ~unsafe & self.flags['valid']
        self.pulses['return'] &= ~unsafe & self.flags['success']
