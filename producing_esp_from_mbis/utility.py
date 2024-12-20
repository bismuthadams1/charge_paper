from openff.units import unit
import numpy as np

AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge

def assign_esp(monopoles : np.ndarray[float],
               dipoles : np.ndarray[float],
               quadropules : np.ndarray[float],
               grid: unit.Quantity,
               coordinates: unit.Quantity) -> list[float,float]:
    """Assign charges according to charge model selected

    Parameters
    ----------
    ob_mol: generic python object depending on the charge model
        Charge model appropriate python object on which to assign the charges

    Returns
    -------
    partial_charges: list of partial charges 
    """

    #multipoles with correct units
    monopoles_quantity = monopoles.to(unit.e)
    dipoles_quantity = dipoles.to(unit.e*unit.angstrom)
    quadropoles_quantity = quadropules.to(unit.e*unit.angstrom*unit.angstrom)
    coordinates_ang = coordinates.to(unit.angstrom)
    monopole_esp = calculate_esp_monopole_au(grid_coordinates=grid,
                                        atom_coordinates=coordinates_ang,
                                        charges = monopoles_quantity)
    dipole_esp = calculate_esp_dipole_au(grid_coordinates=grid,
                                    atom_coordinates=coordinates_ang,
                                    dipoles= dipoles_quantity)
    quadrupole_esp = calculate_esp_quadropole_au(grid_coordinates=grid,
                                    atom_coordinates=coordinates_ang,
                                    quadrupoles= quadropoles_quantity)
    #NOTE: ESP units, hartree/e and grid units are angstrom
    return (monopole_esp + dipole_esp + quadrupole_esp).m.flatten().tolist(), grid.m.tolist()


def calculate_esp_monopole_au(
    grid_coordinates: unit.Quantity,  # N x 3
    atom_coordinates: unit.Quantity,  # M x 3
    charges: unit.Quantity,  # M
    ):
    """Generate the esp from the on atom monopole
    
    Parameters
    ----------
    grid_coordinates: unit.Quantity
        grid on which to build the esp on 

    atom_coordinates: unit.Quantity
        coordinates of atoms to build the esp  
    
    charges: unit.Quantity
        monopole or charges

    Returns
    -------
    monopole_esp: unit.Quantity
        monopole esp
    """
    #prefactor
    ke = 1 / (4 * np.pi * unit.epsilon_0) # 1/vacuum_permittivity, 1/(e**2 * a0 *Eh)

    #Ensure everything is in AU and correct dimensions
    charges = charges.flatten()
    grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)  #Å to Bohr
    atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)    #Å to Bohr
    #displacement and distance
    displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N x M x 3 B
    distance = np.linalg.norm(displacement.m, axis=-1)*unit.bohr # N, M
    inv_distance = 1 / distance  #N, M

    esp = ke*np.sum(inv_distance * charges[None,:], axis=1)  # (N,M)*(1,M) -> (N,M) numpy broadcasts all charges. Over all atoms  =  Sum over M (atoms), resulting shape: (N,) charges broadcast over each N
    # print('monopole esp')
    # print(esp)
    return esp.to(AU_ESP)

def calculate_esp_dipole_au(
    grid_coordinates: unit.Quantity,  # N , 3
    atom_coordinates: unit.Quantity,  # M , 3
    dipoles: unit.Quantity,  # M , 3       
    ) -> unit.Quantity:
    """Generate the esp from the on atom dipoles
    
    Parameters
    ----------
    grid_coordinates: unit.Quantity
        grid on which to build the esp on 

    atom_coordinates: unit.Quantity
        coordinates of atoms to build the esp  
    
    dipoles: unit.Quantity
        dipoles or charges

    Returns
    -------
    dipoles_esp: unit.Quantity
        monopole esp
    """

    #prefactor
    ke = 1 / (4 * np.pi * unit.epsilon_0) # 1/vacuum_permittivity, 1/(e**2 * a0 *Eh)

    #Ensure everything is in AU
    dipoles = dipoles.to(unit.e*unit.bohr)
    grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)  #Å to Bohr
    atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)    #Å to Bohr

    displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N , M , 3 
    distance = np.linalg.norm(displacement.m, axis=-1)*unit.bohr # N, M 
    inv_distance_cubed = 1 / distance**3 #1/B
    #Hadamard product for element-wise multiplication
    dipole_dot = np.sum(displacement * dipoles[None,:,:], axis=-1) # dimless * e.a

    esp = ke*np.sum(inv_distance_cubed* dipole_dot,axis=1) # e.a/a**2 
    # print('dipole esp')
    # print(esp)
    return esp.to(AU_ESP)

def calculate_esp_quadropole_au(
    grid_coordinates: unit.Quantity,  # N x 3
    atom_coordinates: unit.Quantity,  # M x 3
    quadrupoles: unit.Quantity,  # M 3 3
    ) -> unit.Quantity:
    """Generate the esp from the on atom quandropoles
    
    Parameters
    ----------
    grid_coordinates: unit.Quantity
        grid on which to build the esp on 

    atom_coordinates: unit.Quantity
        coordinates of atoms to build the esp  
    
    quandropoles: unit.Quantity
        dipoles or charges

    Returns
    -------
    quandropoles_esp: unit.Quantity
        monopole esp
    """

    #prefactor
    ke = 1 / (4 * np.pi * unit.epsilon_0) # 1/vacuum_permittivity, 1/(e**2 * a0 *Eh)
    #Ensure everything is in AU
    quadrupoles = detrace(quadrupoles.to(unit.e*unit.bohr*unit.bohr))

    grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)  #Å to Bohr
    atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)    #Å to Bohr

    displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N , M , 3 
    distance = np.linalg.norm(displacement.m, axis=-1)*unit.bohr # N, M 
    inv_distance = 1 / distance #1/B

    quadrupole_dot_1 = np.sum(quadrupoles[None,:,:] * displacement[:,:,None],axis=-1)
    quadrupole_dot_2 = np.sum(quadrupole_dot_1*displacement,axis=-1)
    esp = ke*np.sum((3*quadrupole_dot_2*(1/2 * inv_distance**5)),axis=-1)
    # print('quadrupole esp')
    # print(esp)
    return esp.to(AU_ESP)

def detrace(quadrupoles: unit.Quantity) -> unit.Quantity:

    #convert to numpy array
    quadrupoles = quadrupoles.m
    for i in range(quadrupoles.shape[0]):
        trace = np.trace(quadrupoles[i])
        trace /= 3
        # make sure we have the traceless quad tensor

        quadrupoles[i][0][0]=quadrupoles[i][0][0] - trace
        quadrupoles[i][0][1]=quadrupoles[i][0][1]
        quadrupoles[i][0][2]=quadrupoles[i][0][2]
        quadrupoles[i][1][1]=quadrupoles[i][1][1] - trace
        quadrupoles[i][2][2]=quadrupoles[i][2][2] - trace
        quadrupoles[i][1][2]=quadrupoles[i][1][2]

    #convert back to unit quantity
    quadrupoles = quadrupoles*unit.e*unit.bohr*unit.bohr
    return quadrupoles