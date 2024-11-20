import sys
import time
import math
from random import randint, random
from typing import Dict, Any, Union, Literal

import numpy as np
from virtaccl.site.SNS_Linac.virtual_devices import Cavity, Quadrupole, Corrector, WireScanner

from virtaccl.beam_line import Device, AbsNoise, LinearT, PhaseT, PhaseTInv, LinearTInv, PosNoise

# Here are the device definitions that take the information from PyORBIT and translates/packages it into information for
# the server.
#
# All the devices need a name that will determine the name of the device in the EPICS server. If the corresponding
# element in PyORBIT has a different name, then it will need to be specified in the declaration as the model_name. If
# the device has settings (values changed by the user), they can be given initial values to match the model using
# initial_dict. These initial settings need to be in a dictionary using the appropriate keys defined by PyORBIT to
# differentiate different parameters. And finally, if the device has a phase, both setting and measurement, they can be
# given an offset using phase_offset.
#
# The strings denoted with a "_pv" are labels for EPICS. Changing these will alter the EPICS labels for values on the
# server. The strings denoted with a "_key" are the keys for parameters in PyORBIT for that device. These need to match
# the keys PyORBIT uses in the paramsDict for that devices corresponding PyORBIT element.

class BTF_Actuator(Device):
    # EPICS PV names
    position_set_pv = 'DestinationSet' #[mm]
    position_sync_readback_pv = 'PositionSync' #[mm]
    position_enc_readback_pv = 'PositionEnc' #[mm]

    velocity_set_pv = 'VelocitySet' #[mm/s]

    command_set_pv = 'Command'
    status_readback_pv = 'Status'

    # Device keys
    position_key = 'position' #[m]

    def __init__(self, name: str, model_name: str, park_location = None, velocity = None, velocity_limit = None, pos_limit = None):
        self.model_name = model_name
        super().__init__(name, self.model_name)

        # Changes the units from meters to millimeters for associated PVs
        self.milli_units = LinearTInv(scaler=1e3)

        # Creates flat noise
        pos_noise = AbsNoise(noise=1e-3)

        # Sets initial values for parameters
        if park_location is not None:
            self.park_location = park_location
        else:
            self.park_location = -0.07

        if velocity is not None:
            self.velocity = velocity
        else:
            self.velocity = 0.0015

        if velocity_limit is not None:
            self.velocity_limit = velocity_limit
        else:
            self.velocity_limit = 0.01

        if pos_limit is not None:
            self.pos_limit = pos_limit
        else:
            self.pos_limit = -0.016

        initial_command = -1 # Stasis value that does not effect actuator movement
        initial_position = self.park_location
        initial_velocity = self.velocity

        # Defines internal parameters to keep track of the screen position
        self.last_actuator_pos = initial_position
        self.last_actuator_time = time.time()
        self.command_value = initial_command
        self.pos_goal = initial_position

        # Registers the device's PVs with the server
        self.register_setting(BTF_Actuator.position_set_pv, default = initial_position, transform = self.milli_units)
        self.register_readback(BTF_Actuator.position_sync_readback_pv, BTF_Actuator.position_set_pv, transform = self.milli_units, noise=pos_noise)
        self.register_readback(BTF_Actuator.position_enc_readback_pv, BTF_Actuator.position_set_pv, transform = self.milli_units, noise=pos_noise)

        self.register_setting(BTF_Actuator.velocity_set_pv, default = initial_velocity, transform = self.milli_units)

        self.register_setting(BTF_Actuator.command_set_pv, default = initial_command, definition={'type': 'int'})
        self.register_readback(BTF_Actuator.status_readback_pv, BTF_Actuator.command_set_pv, definition={'type': 'int'})

    def get_actuator_position(self):
        # Initialize last position and time
        last_pos = self.last_actuator_pos
        last_time = self.last_actuator_time

        # Update position goal if command status is changed
        command_status = self.get_parameter_value(BTF_Actuator.command_set_pv)

        if command_status == 1:
            self.pos_goal = last_pos
        elif command_status == 2:
            self.pos_goal = self.get_parameter_value(BTF_Actuator.position_set_pv)
        elif command_status == 3:
            self.pos_goal = self.park_location

        # reset command status to stasis value
        if command_status != -1:
            self.server_setting_override(BTF_Actuator.command_set_pv,-1)

        # Adjust final position if it is outside actuator bounds
        final_pos = self.pos_goal
        park_loc = self.park_location
        pos_lim = self.pos_limit

        if park_loc < pos_lim:
            if final_pos > pos_lim:
                final_pos = pos_lim
            elif final_pos < park_loc:
                final_pos = park_loc
        elif park_loc > pos_lim:
            if final_pos < pos_lim:
                final_pos = pos_lim
            elif final_pos > park_loc:
                final_pos = park_loc

        # Update position goal to match final position
        self.pos_goal = final_pos

        # Limit the velocity of the actuator to the maximum velocity of the physical actuator
        actuator_velocity = self.get_parameter_value(BTF_Actuator.velocity_set_pv)
        vel_lim = self.velocity_limit

        if actuator_velocity > vel_lim:
            actuator_velocity = vel_lim
            self.set_parameter_value(BTF_Actuator.velocity_set_pv,actuator_velocity)

        current_time = time.time()
        current_pos = self.last_actuator_pos
        if final_pos != last_pos:
            direction = np.sign(final_pos - last_pos)
            current_pos = direction * actuator_velocity * (current_time - last_time) + last_pos

            if direction < 0 and current_pos < final_pos:
                current_pos = final_pos
            elif direction > 0 and current_pos > final_pos:
                current_pos = final_pos

        # reset variables for the next calculation
        self.last_actuator_time = current_time
        self.last_actuator_pos = current_pos

        return current_pos

    # Return the setting value of the PV name for the device as a dictionary using the model key and it's value.
    # This is where the setting PV names are associated with their model keys
    def get_model_optics(self) -> Dict[str, Dict[str, Any]]:
        actuator_position = self.last_actuator_pos

        params_dict = {BTF_Actuator.position_key: actuator_position}
        model_dict = {self.model_name: params_dict}
        return model_dict

    # For the input setting PV (not the readback PV), updates it's associated readback on the server using the model
    def update_readbacks(self):
        actuator_position = BTF_Actuator.get_actuator_position(self)
        self.update_readback(BTF_Actuator.position_sync_readback_pv, actuator_position)
        self.update_readback(BTF_Actuator.position_enc_readback_pv, actuator_position)

        status_names = [0,1,2]

        position_goal = self.pos_goal
        park_position = self.park_location

        if actuator_position == position_goal and actuator_position != park_position:
            current_status = status_names[0]
        elif actuator_position == park_position:
            current_status = status_names[2]
        else:
            current_status = status_names[1]

        self.update_readback(BTF_Actuator.status_readback_pv, current_status)

class BTF_FC(Device):
    #EPICS PV names
    current_pv = 'CurrentAvrGt' # [mA]
    state_set_pv = 'State_Set'
    state_readback_pv = 'State'
    current_noise = -7e-2

    #PyORBIT parameter keys
    current_key = 'current'
    state_key = 'state'

    def __init__(self, name: str, model_name: str = None, init_state=None):

        self.model_name = model_name
        super().__init__(name, self.model_name)

        # Changes the units from meters to millimeters for associated PVs.
        self.milli_units = LinearTInv(scaler=1e3)

        current_noise = PosNoise(noise=BTF_FC.current_noise)

        # Registers the device's PVs with the server
        self.register_measurement(BTF_FC.current_pv, noise=current_noise, transform = self.milli_units)

        self.register_setting(BTF_FC.state_set_pv, default=init_state)
        self.register_readback(BTF_FC.state_readback_pv, BTF_FC.state_set_pv)

    # Return the setting value of the PV name for the device as a dictionary using the model key and it's value. This is
    # where the PV names are associated with their model keys.
    def get_model_optics(self) -> Dict[str, Dict[str, Any]]:
        new_state = self.get_parameter_value(BTF_FC.state_set_pv)

        params_dict = {BTF_FC.state_key: new_state}
        model_dict = {self.model_name: params_dict}
        return model_dict

    def update_readbacks(self):
        fc_state = self.get_parameter_value(BTF_FC.state_set_pv)
        self.update_readback(BTF_FC.state_readback_pv, fc_state)

    # Updates the measurement values on the server. Needs the model key associated with its value and the new value.
    # This is where the measurement PV name is associated with it's model key.
    def update_measurements(self, new_params: Dict[str, Dict[str, Any]] = None):
        current_state = self.get_parameter_value(BTF_FC.state_set_pv)

        if current_state == 1:
            fc_params = new_params[self.model_name]
            current = -1.0*fc_params[BTF_FC.current_key]
        else:
            current = 0
        self.update_measurement(BTF_FC.current_pv, current)


class BTF_BCM(Device):
    #EPICS PV names
    current_pv = 'CurrentAvrGt' # [mA]
    current_noise = -7e-2

    #PyORBIT parameter keys
    current_key = 'current'

    def __init__(self, name: str, model_name: str = None):
        if model_name is None:
            self.model_name = name
        else:
            self.model_name = model_name
        super().__init__(name, self.model_name)

        # Changes the units from meters to millimeters for associated PVs.
        self.milli_units = LinearTInv(scaler=1e3)

        current_noise = PosNoise(noise=BTF_BCM.current_noise)

        # Registers the device's PVs with the server
        self.register_measurement(BTF_BCM.current_pv, noise=current_noise, transform=self.milli_units)

    # Updates the measurement values on the server. Needs the model key associated with its value and the new value.
    # This is where the measurement PV name is associated with it's model key.
    def update_measurements(self, new_params: Dict[str, Dict[str, Any]] = None):
        bcm_params = new_params[self.model_name]
        current = -1.0*bcm_params[BTF_BCM.current_key]
        self.update_measurement(BTF_BCM.current_pv, current)

class BTF_Quadrupole(Device):
    # EPICS PV names
    field_readback_pv = 'B' # [T/m]
    field_noise = 1e-6 # [T/m]

    # PyORBIT parameter keys
    field_key = 'dB/dr'

    def __init__ (self, name: str, model_name: str, power_supply: Device, coeff_a=None, coeff_b=None, length=None):

        self.model_name = model_name
        self.power_supply = power_supply
        self.coeff_a = coeff_a
        self.coeff_b = coeff_b
        self.length = length

        connected_devices = power_supply

        super().__init__(name, self.model_name, connected_devices)

        field_noise = AbsNoise(noise=BTF_Quadrupole.field_noise)

        # Registers the device's PVs with the server
        self.register_readback(BTF_Quadrupole.field_readback_pv, noise=field_noise)

    # Return the setting value of the PV name for the device as a dictionary using the model key and it's values.
    # This is where the PV names are associated with their model keys.
    def get_current_from_PS(self):
        new_current = self.power_supply.get_parameter_value(BTF_Quadrupole_Power_Supply.current_set_pv)
        sign = np.sign(new_current)
        new_current = np.abs(new_current)

        GL = sign*(self.coeff_a*new_current + self.coeff_b*new_current**2)

        new_field = - GL/self.length

        if self.model_name == 'MEBT:QV02':
            new_field = -new_field

        return new_field

    def get_model_optics(self) -> Dict[str, Dict[str, Any]]:
        new_field = self.get_current_from_PS()

        params_dict = {BTF_Quadrupole.field_key: new_field}
        model_dict = {self.model_name: params_dict}
        return model_dict

    def update_readbacks(self):
        rb_field = self.get_current_from_PS()
        self.update_readback(BTF_Quadrupole.field_readback_pv, rb_field)



class BTF_Quadrupole_Power_Supply(Device):
    current_set_pv = 'I_Set' # [Amps]
    current_readback_pv = 'I' # [Amps]

    def __init__(self, name: str, init_current=None):
        super().__init__(name)

        field_noise = AbsNoise(noise=1e-6)

        self.register_setting(BTF_Quadrupole_Power_Supply.current_set_pv, default=init_current)
        self.register_readback(BTF_Quadrupole_Power_Supply.current_readback_pv, BTF_Quadrupole_Power_Supply.current_set_pv)

class BTF_Corrector(Device):
    # EPICS PV names
    field_readback_pv = 'B'  # [T]
    field_noise = 1e-6  # [T/m]

    # PyORBIT parameter keys
    field_key = 'B'  # [T]

    def __init__ (self, name: str, model_name: str, power_supply: Device, coeff=None, length=None, momentum=None):

        self.model_name = model_name
        self.power_supply = power_supply
        self.coeff = coeff
        self.length = length
        self.momentum = momentum

        super().__init__(name, self.model_name, self.power_supply)

        field_noise = AbsNoise(noise=BTF_Corrector.field_noise)

        # Registers the device's PVs with the server
        self.register_readback(BTF_Corrector.field_readback_pv, noise=field_noise)

    def get_current_from_PS(self):
        new_current = self.power_supply.get_parameter_value(BTF_Corrector_Power_Supply.current_set_pv)

        new_field = (self.coeff * 1e-3 * new_current * self.momentum) / (self.length * 0.299792)

        return new_field

    def get_model_optics(self) -> Dict[str, Dict[str, Any]]:
        new_field = self.get_current_from_PS()

        params_dict = {BTF_Corrector.field_key: new_field}
        model_dict = {self.model_name: params_dict}
        return model_dict

    def update_readbacks(self):
        rb_field = self.get_current_from_PS()
        self.update_readback(BTF_Corrector.field_readback_pv, rb_field)

class BTF_Corrector_Power_Supply(Device):
    current_set_pv = 'I_Set' # [Amps]
    current_readback_pv = 'I' # [Amps]

    def __init__(self, name: str, init_current=None):
        super().__init__(name)

        field_noise = AbsNoise(noise=1e-6)

        self.register_setting(BTF_Corrector_Power_Supply.current_set_pv, default=init_current)
        self.register_readback(BTF_Corrector_Power_Supply.current_readback_pv, BTF_Corrector_Power_Supply.current_set_pv)

