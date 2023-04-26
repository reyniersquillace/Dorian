import numpy as np
import pandas as pd

class Molecule():
    ''' 
    This class describes a molecule. Its properties can be entered and can allow for any molecule
    as long as that molecule happens, by coincidence, to be either 14NH2D or 15NH2D. It takes one
    energy transition as an argument.
    '''
    
    def __init__(self, file_name = str, symmetry = 'ortho', ortho_g = 3, para_g = 1, 
                 relative_intensity = 1):

        #file_name is the name of the files containing the relevant energy levels
        self.file_name = file_name

        #ortho or para
        self.symmetry = symmetry
        if symmetry == 'ortho':
            self.g = ortho_g
        if symmetry == 'para':
            self.g = para_g
        self.R = relative_intensity


    def Partition(self, T):
        ''' 
        This function evaluates the Partition function of a given molecule for a given range of
        energy levels included in the levels file. The levels file for a non-hyperfine structure
        must be formatted with five columns separated by commas (Eu/k, J, Ka, Kc, Symmetry). The 
        levels file for a hyperfine structure must be formatted similarly with the columns (Eu/k, 
        Ka, Kc, F1, F). 

        Inputs:
        -------
                self

        Returns:
        --------
                Q (float or DataFrame): sum(g*exp(-Eu/kT))
                T (float or array)    : temperature at which to evaluate Partition function

        '''

        levels_file = open(self.file_name,'r')

        #for non-hyperfine transitions:
        if self.R == 1:
            #the columns in the data file:
            idlist = ['E/k', 'J', 'Ka', 'Kc', 'Sym']
            df = pd.read_csv(levels_file, names = idlist)
            levels_file.close()
            ortho_df = pd.DataFrame(columns = idlist)
            para_df = pd.DataFrame(columns = idlist)

            for i in range(0, len(df)):
            #the molecule is ortho if sym+Ka is odd, and para if even
                if (df.at[i, 'Sym']+df.at[i, 'Ka'])%2!=0:
                    ortho_df = pd.concat([ortho_df, pd.DataFrame([df.iloc[i].to_list()], columns = idlist)])
                else:
                    para_df = pd.concat([para_df, pd.DataFrame([df.iloc[i].to_list()], columns = idlist)])
            ortho_df = ortho_df.set_index(pd.Index(range(0, len(ortho_df))))
            para_df = para_df.set_index(pd.Index(range(0, len(para_df))))

            if self.symmetry == 'ortho':
                df = ortho_df
            elif self.symmetry == 'para':
                df = para_df
            else:
                raise Exception('Whoops! You sent in a non-valid symmetry.')

        #for hypefine transitions, the levels file already consists only of ortho transitions
        else:
            idlist = ['E/k', 'J', 'Ka', 'Kc', 'F1', 'F']
            df = pd.read_csv(levels_file, names = idlist)
            levels_file.close()
        
        def partition_func(df, i, T):
            ''' 
            This function performs the mathematical evaluation of one level that contributse to the total 
            partition function at a given temperature.

            Inputs:
            -------
                    df (DataFrame): contains the information in the levels file
                    i (int)       : index of the energy level in question
                    T (K)         : excitation temperature

            Returns:
            --------
                    q (float)     : gu*exp(-E/kT)
            '''

            if self.R == 1:
                if type(T) == u.quantity.Quantity:
                    T = T.value
                gu = 3*(2*df.at[i, 'J'] + 1)
                return gu*np.exp(-df.at[i, 'E/k']/T)
            else:
                if type(T) == u.quantity.Quantity:
                    T = T.value
                gu = 2*df.at[i, 'F'] + 1
                return gu*np.exp(-df.at[i, 'E/k']/T)

        #if the exact excitation temperature is unknown, iterate through range
        if type(T) != u.quantity.Quantity:
            Q = []
            #sum together all individual contributions!
            for Ttemp in T:
                Q.append(np.sum(np.array([partition_func(df, i, Ttemp) for i in range(0, len(df))])))
            Q_df = pd.DataFrame({'Q(Tex)': Q})
            Q_df = Q_df.set_index(pd.Index(T))
            return Q_df

        #otherwise return discrete value
        else:
            Q = (np.sum(np.array([partition_func(df, i, T) for i in range(0, len(df))])))
            return Q 
        
import astropy.units as u
import astropy.constants as const
import numpy as np
import pandas as pd

class Observation():
    
    def __init__(self, nu, TA_star, v = 0, Tex = ('unknown', 'calculate', float), filename = None, Tex_min = 2.8, 
                Tex_max = 10, dT = 0.05, vlsr = None , dv = None, tau_main = None, 
                 eff_MB = 1, fill_frac = 0.5, Tbg = 2.725*u.K, Eu_k = None, dipole = None, 
                 Einstein_A = None, S = 1.5, Int_R = 1):
        '''
                nu (Hz)          : the frequency of the line 
                TA_star (K)      : the observed intensity in temperature units
                v (km/s)         : the central velocity of the line
                Tex (str)        : the excitation temperature, if known
                filename (str)   : the name of the file containing the hyperfine observations:
                                   <relative intensity>,<velocity of the line>
                Tex_min (float)  : minimum value of excitation temperature
                Tex_max (float)  : maximum value of excitation temperature
                dT (float)       : interval for varying excitation temperature
                vlsr (km/s)      : the velocity of the local standard of rest
                dv (km/s)        : the FWHM of the observation
                opt_depth (float): the optical depth of the medium
                Tbg (K)          : the background temperature
                eff_MB (float)   : the main beam efficiency of the telescope
                fill_frac (float): the filling fraction of the telescope
                Tbg (K)          : the background temperature
                Eu_k (K)         : the upper energy level divided by Boltzmann's constant
                dipole (float)   : the electric dipole moment of the transition
                Einstein_A (s^-1): the Einstein A of the transition in question
                S (float)        : the symmetry of the rotor 
                Int_R (float)    : the relative intensity of the hyperfine structure
        '''

        self.nu = nu
        self.I = TA_star/(eff_MB*fill_frac)   
        self.v = v
        self.vlsr = vlsr
        self.dv = dv 
        self.tau_main = tau_main
        self.Tbg = Tbg
        self.Eu_k = Eu_k
        self.dipole = dipole
        self.A = Einstein_A
        self.S = S
        self.R = Int_R
        self.filename = filename

        #allow for excitation temperature to be either a range or one discrete number
        if Tex == 'unknown':
            self.Tex = np.array(np.linspace(Tex_min, Tex_max, int((Tex_max - Tex_min)/dT + 1)))
        elif Tex == 'calculate':
            self.Tex = Observation.ExcitationTemperature(self)
        else:
            self.Tex = Tex

        if Int_R != 1:
            self.tau_nu = Observation.OpticalDepth(self)

    def SourceFunction(self, T):
        '''
        This function evaluates the Source Function of a transition at a given temperature.

        Inputs:
        -------
                T (K)  : the  temperature at which to evaluate

        Returns:
        -------
                J (float): h*nu/k * 1/(exp(h*nu/kT) - 1)

        '''
        num = ((const.h.cgs*self.nu)/(const.k_B.cgs))
        denom = ((np.exp((const.h.cgs*self.nu)/(const.k_B.cgs*T)))-1)
        J = num/denom
        return J.to(u.K)

    def OpticalDepth(self):
        ''' 
        This function calculates the optical depth at a given line from the optical depth derived by 
        observation of the hyperfine structure. 

        Inputs:
        -------
                tau_main (float): the observed optical depth

        Returns:
        --------
                tau_nu (float): the actual optical depth of the observation
        '''

        levels = np.genfromtxt(self.filename, delimiter = ',')

        tau_nu = 0
        for line in levels:
            v_stuff = (self.v-self.vlsr-line[0]*u.km/u.s)
            tau_nu += (line[1])*np.exp((-4*np.log(2))*(v_stuff**2)/(self.dv**2))
        
        tau_nu *= self.tau_main

        return tau_nu


    def ExcitationTemperature(self):
        '''
        This function calculates the excitation temperature of a molecule from observations 
        in an optically thick medium.

        Inputs:
        -------
                filename (str): the name of the file containing the hyperfine observations 
                                formatted: <relative intensity>,<velocity of the line>
        Returns:
        --------
                T_ex (K): the excitation temperature of the molecule in the medium
        '''
        import astropy.constants as const
        import astropy.units as u
        import numpy as np

        hnuk = const.h.cgs*self.nu/const.k_B.cgs
        thing_in_denominator = self.I/((1-np.exp(-Observation.OpticalDepth(self))))
        whole_denominator = hnuk/(thing_in_denominator 
                                  + Observation.SourceFunction(self, self.Tbg)) + 1
        T_ex = hnuk/np.log(whole_denominator)

        return T_ex.to(u.K)

def ColumnDensity(Mol, Obs, opt_thin = False, opt_thick = True):
    ''' 
    This function evaluates the column density of the molecule at a given excitation temperature
    or a range of excitation temperatures.

    Inputs:
    -------
            Mol (Molecule)     : a Molecule object created in Dorian
            Obs (Observation)  : an Observation object created in Iona
            opt_thin (Boolean) : optically thin? 
            opt_thick (Boolean): optically thin? 


    Returns:
    --------
            N (float or DataFrame): the column density of the molecule
    '''

    def exp_term(T):
        '''
        This function evaluates the exponential term within column density calculations at a given 
        temperature. 

        Inputs:
        -------
                T (K): the excitation temperature

        Returns:
                exp(Eu/kT)/(exp(h*nu/kT)-1)
        '''
        num = np.exp(Obs.Eu_k/T)
        denom = ((np.exp(const.h.cgs*Obs.nu/(const.k_B.cgs*T)))-1)
        return (num/denom).to_value()

    if opt_thin:
        #9 is coming from stat weight of upper level (g = 3*(2J+1) for ortho)
        #the following is the long constant term at the beginning of the optically thin N calculation
        #took out 9 in denominator
        
        C = (3*const.h.cgs.value)/(8*(np.pi**3)*(Obs.S)*((Obs.dipole)**2)*Obs.R*Mol.g)

        #evaluate the source function at the background temperature
        J_tbg = Obs.SourceFunction(T = Obs.Tbg)

        def J_term(T):
            ''' 
            This function evaluates the source function term in the column density calculation.
            
            Inputs:
            -------
                    T (float, array): excitation temperature

            Returns:
            --------
                    1/(J(T) - J(T_bg))
            '''
            return 1/((Obs.SourceFunction(T = T))-J_tbg)

        #case where we have a range of possible excitation temperatures
        if type(Obs.Tex) == np.ndarray:
            #evaluate the partition function at each possible T
            Q = np.array((Mol.Partition(T = Obs.Tex))['Q(Tex)'].to_list())
            N = np.empty(len(Q))
            #evaluate the column density at each possible T
            for i in range(len(Q)):
                N[i] = ((Q[i])*exp_term(Obs.Tex[i]*u.K)*J_term(Obs.Tex[i]*u.K).to_value())
            N = Obs.I*C*N*1e5
            df = pd.DataFrame({'N(Tex)': N})
            df = df.set_index(Obs.Tex)
            return df

        #case where we know the excitation temperature
        else:
            Q = Mol.Partition(T = Obs.Tex)
            N = ((Q)*exp_term(Obs.Tex)*J_term(Obs.Tex))
            N = Obs.I*C*N*1e5
            return N
        
    #optically thick case  
    elif opt_thick:
        #evaluate constant terms at beginning of formula
        c1=(np.pi/(4*np.log(2)))**0.5 #sqrt(pi/4ln2)
        c2=8*np.pi*(Obs.nu**3)/(Obs.A*Mol.g*(const.c.cgs**3)) #8*pi*nu^3/(A*g*c^3)

        #evaluate over range of Ts if T unknown
        if type(Obs.Tex) == np.ndarray:
            Q = np.array((Mol.Partition(T = Obs.Tex))['Q(Tex)'].to_list())
            N = np.empty(len(Q))
            for i in range(len(Q)):
                N[i] = exp_term(Obs.Tex[i])*Q[i]
            N = Obs.R*Obs.tau_main*Obs.dv.to(u.cm/u.s)*c1*c2*N
            df = pd.DataFrame({'N(Tex)': N})
            df = df.set_index(Obs.Tex)
            return df

        else:
            Q = Mol.Partition(T = Obs.Tex)
            N = exp_term(Obs.Tex)*Q
            N = Obs.R*Obs.tau_main*Obs.dv.to(u.cm/u.s)*c1*c2*N
            return Obs.Tex, N.to(u.cm**-2)

