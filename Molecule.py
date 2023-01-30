import numpy as np
import pandas as pd

#needs generalization-- mess with S and Tbg if needed. Will be improved in next push.
h=6.6260755e-27
S=0.5
k=1.3807e-16
Tbg=2.725
c = 2.9979245e10

class Molecule():
    ''' 
    This class describes a molecule. Its properties can be entered and can allow for any molecule
    as long as that molecule happens, by coincidence, to be either 14NH2D or 15NH2D. It takes one
    energy transition as an argument.
    '''
    
    def __init__(self, file_name = str, symmetry = 'ortho', Tex = ('unknown', float), Tex_min = 2.8, 
                Tex_max = 10, dT = 0.05, ortho_g = 3, para_g = 1, relative_intensity = 1):

        #allow for excitation temperature to be either a range or one discrete number
        if Tex == 'unknown':
            self.Tex = np.array(np.linspace(Tex_min, Tex_max, int((Tex_max - Tex_min)/dT + 1)))
        else:
            self.Tex = Tex

        #file_name is the name of the files containing the relevant energy levels
        self.file_name = file_name

        #ortho or para
        self.symmetry = symmetry
        if symmetry == 'ortho':
            self.g = ortho_g
        if symmetry == 'para':
            self.g = para_g

        #relative intensity of peak transition in hyperfine structure
        self.ri = relative_intensity

    def Partition(self):
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

        '''

        levels_file = open(self.file_name,'r')

        #for non-hyperfine transitions:
        if self.ri == 1:
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
                    T (float)     : excitation temperature

            Returns:
            --------
                    q (float)     : gu*exp(-E/kT)
            '''

            if self.ri == 1:
                gu = 3*(2*df.at[i, 'J'] + 1)
            else:
                gu = 2*df.at[i, 'F'] + 1
            return gu*np.exp(-df.at[i, 'E/k']/T)

        #if the exact excitation temperature is unknown, iterate through range
        if type(self.Tex) != float:
            Q = []
            #sum together all individual contributions!
            for T in self.Tex:
                Q.append(np.sum(np.array([partition_func(df, i, T) for i in range(0, len(df))])))
            Q_df = pd.DataFrame({'Q(Tex)': Q})
            Q_df = Q_df.set_index(pd.Index(self.Tex))
            return Q_df

        #otherwise return discrete value
        else:
            Q = (np.sum(np.array([partition_func(df, i, self.Tex) for i in range(0, len(df))])))
            return Q 
    
    def SourceFunction(self, T, nu):
        '''
        This function evaluates the Source Function of a transition at a given temperature.

        Inputs:
        -------
                self
                T (float): the excitation temperature

        Returns:
        -------
                J(T) = h*nu/k * 1/(exp(h*nu/kT) - 1)

        '''
        return ((h*nu)/(k))/((np.exp((h*nu)/(k*T)))-1)

    def ColumnDensity(self, nu, Eu_k, opt_depth = 1., opt_thin = False, opt_thick = True,
                    int_intensity = 1., fill_frac = 0.5, Tbg = 2.725, dv = None, dipole = None,
                    Einstein_A = None):
        ''' 
        This function evaluates the column density of the molecule at a given excitation temperature
        or a range of excitation temperatures.

        Inputs:
        -------
                self
                nu (float)            : the peak frequency of the given transition in GHz
                Eu_k (float)          : the upper energy divided by Bolztmann's constant
                opt_depth (float)     : the optical depth of the given transition, if relevant
                opt_thin (Boolean)    : optically thin? 
                opt_thick (Boolean)   : optically thin? 
                int_intensity (float) : the integrated intensity of the observed spectrum
                fill_frac (float)     : the filling fraction of the object in the telescope
                Tbg (float)           : the background temperature
                dv (float)            : the FWHM of the spectrum, if relevant
                dipole (float)        : the electric dipole moment of the transition
                Einstein_A (float)    : the Einstein A of the transition in question

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
                    T (float): the excitation temperature

            Returns:
                    exp(Eu/kT)/(exp(h*nu/kT)-1)
            '''
            return np.exp(Eu_k/T)/((np.exp(h*nu/(k*T)))-1)

        #we need to multiply by 100000 for the units to be consistent.
        units = 1e5

        if opt_depth < 1 or opt_thin:
            #9 is coming from stat weight of upper level (g = 3*(2J+1) for ortho)
            #the following is the long constant term at the beginning of the optically thin N calculation
            C = (3*h)/(8*(np.pi**3)*S*(dipole**2)*self.ri*9)
            #evaluate the source function at the background temperature
            J_tbg = Molecule.SourceFunction(self, Tbg, nu)

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
                return 1/((Molecule.SourceFunction(self, T, nu))-J_tbg)

            #case where we have a range of possible excitation temperatures
            if type(self.Tex) != float:
                #evaluate the partition function at each possible T
                Q = np.array((Molecule.Partition(self))['Q(Tex)'].to_list())
                N = np.empty(len(Q))
                #evaluate the column density at each possible T
                for i in range(len(Q)):
                    N[i] = ((Q[i])*exp_term(self.Tex[i])*J_term(self.Tex[i]))
                N = units*int_intensity*fill_frac*C*N
                df = pd.DataFrame({'N(Tex)': N})
                df = df.set_index(self.Tex)
                return df

            #case where we know the excitation temperature
            else:
                Q = Molecule.Partition(self)
                N = ((Q)*exp_term(self.Tex)*J_term(self.Tex))
                N = units*int_intensity*fill_frac*C*N
                return "{:e}".format(N)
        #optically thick case  
        else:
            #evaluate constant terms at beginning of formula
            c1=(np.pi/(4*np.log(2)))**0.5 #pi/sqrt(4ln2)
            c2=8*np.pi*(nu**3)/(Einstein_A*self.g*(c**3)) #8*pi*nu^3/(A*g*c^3)

            #evaluate over range of Ts if T unknown
            if type(self.Tex) != float:
                Q = np.array((Molecule.Partition(self))['Q(Tex)'].to_list())
                N = np.empty(len(Q))
                for i in range(len(Q)):
                    N[i] = exp_term(self.Tex[i])*Q[i]
                N = units*self.ri*opt_depth*dv*c1*c2*fill_frac*N
                df = pd.DataFrame({'N(Tex)': N})
                df = df.set_index(self.Tex)
                return df

            else:
                Q = Molecule.Partition(self)
                N = exp_term(self.Tex)*Q
                N = units*self.ri*opt_depth*dv*c1*c2*fill_frac*N
                return "{:e}".format(N)