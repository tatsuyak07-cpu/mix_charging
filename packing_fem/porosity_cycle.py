"""Displacement increments to a measured porosity, hold, then zero-load recovery.

Only axial displacement is prescribed during compaction; transverse macro
stresses remain free. Pressure is measured, never capped. The target is the
loaded, equilibrated porosity; recovery porosity is reported separately.
"""
from copy import deepcopy
import math


def smooth(s):
    s=min(1.,max(0.,s))
    return 10*s**3-15*s**4+6*s**5,30*s**2-60*s**3+30*s**4,60*s-180*s**2+120*s**3


class PorosityCycle:
    def __init__(self,c,state=None):
        self.c=c
        required=('target_final_porosity','porosity_tolerance','strain_increment','minimum_strain_increment',
                  'maximum_axial_compression','ramp_s','hold_s','maximum_hold_s','unload_s','recovery_hold_s','maximum_cycle_time_s')
        if any(not math.isfinite(c.get(k,float('nan'))) or c[k]<=0 for k in required):raise ValueError('Positive finite porosity-cycle parameters required')
        if not 0<c['target_final_porosity']<1 or not 0<c['maximum_axial_compression']<.9:raise ValueError('Invalid porosity/strain envelope')
        if not c['minimum_strain_increment']<=c['strain_increment'] or c['maximum_hold_s']<max(c['hold_s'],c['recovery_hold_s']):raise ValueError('Invalid cycle increment/hold budgets')
        self.state=deepcopy(state) if state is not None else dict(phase='ramp',start=0.,start_compression=0.,
            target_compression=min(c['strain_increment'],c['maximum_axial_compression']),
            records=[],loaded_target=False,loaded_porosity=None,loaded_compression=None,peak_force=0.,complete=False,unload_force=None)

    def displacement(self,time):
        s=self.state
        if s['phase']=='ramp':
            T=self.c['ramp_s'];p,v,a=smooth((time-s['start'])/T);d=s['target_compression']-s['start_compression']
            return -(s['start_compression']+d*p),-d*v/T,-d*a/T**2
        return -s['target_compression'],0.,0.

    def interval_end(self):
        s=self.state
        if s['phase']=='ramp':return s['start']+self.c['ramp_s']
        if s['phase']=='unload':return s['start']+self.c['unload_s']
        return float('inf')

    def external_force(self,time):
        s=self.state
        if s['phase']=='unload':return s['unload_force']*(1-smooth((time-s['start'])/self.c['unload_s'])[0])
        return 0.

    def observe(self,time,porosity,compression,axial_force,force_ratio,kinetic_ratio,energy_ratio):
        s=self.state;c=self.c;s['peak_force']=max(s['peak_force'],abs(axial_force))
        if s['phase']=='ramp':
            if time>=s['start']+c['ramp_s']-1e-14:s.update(phase='hold',start=time)
            return None
        if s['phase']=='unload':
            if time>=s['start']+c['unload_s']-1e-14:s.update(phase='recovery',start=time)
            return None
        if s['phase'] not in ('hold','recovery'):return None
        minimum=c['hold_s'] if s['phase']=='hold' else c['recovery_hold_s']
        if time-s['start']<minimum:return None
        checks=dict(force=force_ratio<c['force_tolerance'],kinetic=kinetic_ratio<c['kinetic_tolerance'],energy=energy_ratio<c['energy_tolerance'])
        passed=all(checks.values())
        if not passed and time-s['start']<c['maximum_hold_s']:return None
        record=dict(phase=s['phase'],time_s=time,porosity=porosity,axial_compression=compression,
                    axial_generalized_force=axial_force,force_ratio=force_ratio,kinetic_ratio=kinetic_ratio,energy_ratio=energy_ratio,checks=checks,passed=passed)
        s['records'].append(record)
        if not passed:raise RuntimeError('Porosity-cycle equilibration budget exhausted')
        if s['phase']=='recovery':s.update(phase='complete',complete=True);return 'complete'
        if porosity<=c['target_final_porosity']:
            if porosity<c['target_final_porosity']-c['porosity_tolerance']:raise RuntimeError('Porosity target overshot; replay the preceding accepted stage with a smaller strain increment')
            s.update(loaded_target=True,loaded_porosity=porosity,loaded_compression=compression)
            return 'release'
        if compression>=c['maximum_axial_compression']-1e-12:raise RuntimeError('Axial search envelope exhausted before target porosity; not a physical limit')
        prior=[r for r in s['records'][:-1] if r['phase']=='hold' and r['passed']]
        increment=c['strain_increment']
        if prior:
            previous=prior[-1];delta=compression-previous['axial_compression']
            slope=max((previous['porosity']-porosity)/max(delta,1e-30),0.)
            if slope>0:
                gap=porosity-c['target_final_porosity']
                increment=min(increment,(.25*c['porosity_tolerance'] if gap<4*c['porosity_tolerance'] else .8*gap)/slope)
        increment=max(c['minimum_strain_increment'],increment)
        s.update(phase='ramp',start=time,start_compression=compression,
                 target_compression=min(compression+increment,c['maximum_axial_compression']))
        return 'next_stage'

    def release(self,time,force):
        if not self.state['loaded_target']:raise ValueError('Cannot unload before an equilibrated porosity target')
        self.state.update(phase='unload',start=time,unload_force=force)
