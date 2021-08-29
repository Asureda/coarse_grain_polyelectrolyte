import numpy as np
class propietats:
    def __init__(self, nom = "nom",  sigma= 1., epsilon= 1., Dcoef= 1. , gamma = 'NonDef', carrega = 0, massa = 0.050):
        self.nom = nom
        self.sigma = sigma          # Unitats: nm
        self.epsilon = epsilon      # Unitats: J/mol
        self.gamma = gamma          # Unitats:  [Ener/Dcoeff]
        self.Dcoef = Dcoef          # Unitats: nm2/ns
        self.carrega = carrega
        self.massa = massa          # Unitats: kg/mol

    def calc_gamma(self,TempK):
        R = 8.314472
        if self.Dcoef==0:
            self.gamma=1000000.  # No s'utilitza.
        else:
            self.gamma=(R*TempK)/self.Dcoef  # Escalar o 3 components


# Definir tipus de partícules
#def inicialitza_tipus():

TYPE_HA=0
TYPE_A=1
TYPE_B=2

TYPE_N=3
TYPE_Na=4
TYPE_Cl=5
TYPE_HA2=6
TYPE_A2=7


#Info D:
# https://www.aqion.de/site/diffusion-coefficients
#sigma_POS=0.36   # nm para monovalente
#sigma_POS=0.42  # nm para divalente
#sigma_POS=0.48  # nm para trivalente


#Info D:
# https://www.aqion.de/site/diffusion-coefficients

# Used radius.  sigma=2*radius
#sigma_POS=0.36   # nm para monovalente
#sigma_POS=0.42  # nm para divalente
#sigma_POS=0.48  # nm para trivalente


#Li, Y., Zhang, C., Jiang, Y., Wang, T. J., & Wang, H. (2016). Effects of the hydration ratio on the electrosorption selectivity of ions during capacitive deionization. Desalination, 399, 171–177. https://doi.org/10.1016/j.desal.2016.09.011

#Ion F− Cl− Br− − NO3 2− SO4 Na+ K+ NH4 + Ca2+
#Ion radius, Å 1.36 1.81 1.95 2.64 2.90 0.95 1.33 1.48 0.99
#Hydrated radius, Å 3.52 3.32 3.30 3.35 3.79 3.58 3.31 3.31 4.12

# Avogadro2:  Distance: NH3-NH3 de 3.4 A   sigma=0.34 nm


info_part={TYPE_HA:propietats(nom='HA',sigma=(2*2.6) ,epsilon=1000.0,Dcoef=1.5,carrega=+1.,massa=0.140),
           TYPE_A:propietats(nom='A',sigma=(2*3.172) ,epsilon=1000.0,Dcoef=1.5,carrega=0.,massa=0.140),
           TYPE_B:propietats(nom='B',sigma=(2*2.958) ,epsilon=1000.0,Dcoef=2.2,carrega=+1.,massa=0.140),
           TYPE_N:propietats(nom='N',sigma=(2*3.461) ,epsilon=1000.0,Dcoef=2.0,carrega=0.,massa=0.140),
           TYPE_Na:propietats(nom='Na',sigma=(2*3.9624),epsilon=1000.0,Dcoef=1.33,carrega=+1.,massa=0.140),
           TYPE_Cl:propietats(nom='Cl',sigma=(2*3.915) ,epsilon=1000.0,Dcoef=2.0,carrega=-1.,massa=0.140),
           TYPE_HA2:propietats(nom='HA2',sigma=(2*2.6) ,epsilon=1000.0,Dcoef=0.1,carrega=+1.,massa=0.040),
           TYPE_A2:propietats(nom='A2',sigma=(2*3.172) ,epsilon=1000.0,Dcoef=2.0,carrega=0.,massa=0.017),
           }

Llista_Tipus_Activa=[]

# Calcul de Gamma. PEr Temperature
def calcul_gamma(Tipus_XX,Temperatura):
    info_part[Tipus_XX].calc_gamma(Temperatura)

def calcul_gammas(Temperatura):
    for tipus_i in info_part.keys():
        info_part[tipus_i].calc_gamma(Temperatura)


def comb_epsi( eps1, eps2):
    # rule "Lorentz":
        return (eps1 * eps2)**0.5

def comb_sigma( sig1, sig2):
    # rule "Berthelot":
        return (sig1 + sig2) * 0.5

#def activa(llista_tipus):
#    global Llista_Tipus_Activa
#    Llista_Tipus_Activa=llista_tipus.copy()
#    print('Activant tipus partícules: ',Llista_Tipus_Activa)
    #print('Generant')


def print_info_tipus_Actius():
    print('Llista_Tipus_Activa: ',Llista_Tipus_Activa)
    for ele in Llista_Tipus_Activa:
        print('{:3s}  sigma={} epsilon={} gamma={} Dcoef={} q={:.3f} m={}'.format(info_part[ele].nom,info_part[ele].sigma,
              info_part[ele].epsilon,info_part[ele].gamma,info_part[ele].Dcoef,
              info_part[ele].carrega,info_part[ele].massa))

class dihedres:
    def __init__(self, ll_indices4, syst):
        self.indices4 = ll_indices4  # llista de llistes(de 4 punts)
        self.acum_dih=[]        # angles de -180 a 180
        self.temps=[]
        self.syst=syst


    # Calcul de la llista de dihedres inicialment introduïda a ll_indices4
    def calc(self):
        for ele in self.indices4:
            print('index: {} dihe: {}'.format(ele,self.dihedral2(self.syst.part[ele].pos)))

    def calc1(self,quatreIndex):
        print('index: {} dihe: {}'.format(quatreIndex,self.dihedral2(self.syst.part[quatreIndex].pos)))

    def acum(self):
        dihe_ara=[]
        for ele in self.indices4:
            dihe_ara.append(self.dihedral2(self.syst.part[ele].pos) )

        self.acum_dih.append(dihe_ara)
        #guardem el temps
        self.temps.append(self.syst.time)

        #print(self.acum_dih)

    def get_acum(self):
        return self.acum_dih

    def get_temps(self):
        return self.temps

    def get_index(self):
        return self.indices4

    def save(self,nomfile):
        fn=open(nomfile,'w')
        fn.write('# First column: time\n')
        for ele in self.indices4:
            fn.write('# index_rot: {}\n'.format(ele) )
        for ite in range(len(self.temps)):
            fn.write('{:.4f} '.format(self.temps[ite]) )
            for ele in self.acum_dih[ite]:
                fn.write('{:.2f} '.format(ele))
            fn.write('\n')
        fn.close()


    #https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    # This is the straightforward approach as outlined in the answers to
    # "How do I calculate a dihedral angle given Cartesian coordinates?"
    # p es una llista de 4 coordenades cartesianes.
    def dihedral2(self,p):
        b = p[:-1] - p[1:]
        b[0] *= -1
        v = np.array( [ v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]] ] )
        # Normalize vectors
        v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1,1)
        b1 = b[1] / np.linalg.norm(b[1])
        x = np.dot(v[0], v[1])
        m = np.cross(v[0], b1)
        y = np.dot(m, v[1])
        return np.degrees(np.arctan2( y, x ))

def save_vxyz(system,nomfile,mode='a',aplicar_PBC=True):
    """Guarda coordenades en format xyz (però amb nombre variable de partícules) .vxyz
    Coordenades en angstroms.
    """
    fvxyz=open(nomfile,mode)

    num_particles=len(system.part[:].type)
    #print('num_particles:',num_particles)
    fvxyz.write(str(num_particles)+'\n')
    fvxyz.write('Traj. vxyz\n')

    fcor=10.  # canvi de nm a Angstroms


    #dic_tipus={"HA": 0,"A": 1,"B": 2,"N": 3,"Na": 4,"Cl": 5,"HA2": 6,"A2": 7}
    dic_tipus = {0:'HA',1:'A',2:'B',3:'N',4:'Na',5:'Cl',6:'HA2',7:'A2'}

    for indextp,parttype in enumerate(system.part[:].type):
        part_id=system.part[:].id[indextp]

        xp=system.part[part_id].pos[0]  # Unitats xp: nm
        yp=system.part[part_id].pos[1]
        zp=system.part[part_id].pos[2]

        if aplicar_PBC==True:
            xp=xp%system.box_l[0]   # Unitats xp: nm
            yp=yp%system.box_l[1]
            zp=zp%system.box_l[2]

        # xp=xp*fcor  # Unitats xp: angstroms
        # yp=yp*fcor
        # zp=zp*fcor
        lin='{:3s} {:8.3f} {:8.3f} {:8.3f}'.format(dic_tipus[parttype],xp,yp,zp)
        #print(lin)
        fvxyz.write(lin+'\n')
    fvxyz.close()
    return
def convert_vxyz(nomfile,nom_out):
    global llista_NomsAtoms

    fvxyz=open(nomfile)
    set_NomsAtoms=set([])
    MAX_snaps=9999999
    for snap in range(MAX_snaps):
        #print('Snap',snap)
        lectura=fvxyz.readline()
        if lectura=='':   # Final de fitxer
            max_snapshot=snap-1
            print('Nombre de snapshots:',max_snapshot+1)
            break
        numAt=int(lectura)  # Indica el nombre d'atoms

        lcom=fvxyz.readline()
        llista_NomsAtoms=[]
        for i_at in range(numAt):
            lin=fvxyz.readline()
            words=lin.split()
            llista_NomsAtoms.append(words[0])

        #print('ll',llista_NomsAtoms)

        set_NomsAtoms=set_NomsAtoms|set(llista_NomsAtoms)  #Afegin nous sites
        # Crear un diccionari
    print('Noms dels sites:',set_NomsAtoms)

    fvxyz.close()
    fvxyz=open(nomfile)
    #Identificació del nombre màxim
    dic_max={}
    for snap in range(max_snapshot):
        lectura=fvxyz.readline()
        numAt=int(lectura)
        lcom=fvxyz.readline()
        llista_NomsAtoms=[]
        for i_at in range(numAt):
            lin=fvxyz.readline()
            words=lin.split()
            llista_NomsAtoms.append(words[0])
        dic={}
        for site in set_NomsAtoms:
            dic[site]=llista_NomsAtoms.count(site)

        for ele in dic:
            if not(ele in dic_max):
                dic_max[ele]=dic[ele]
            elif dic_max[ele]<dic[ele]:
                dic_max[ele]=dic[ele]

    print('Total Max. Number',dic_max)

    fvxyz.close()
    # Torna a obrir i generar el nou fitxer

    fvxyz=open(nomfile)
    fout=open(nom_out,'w')

    max_atoms=0
    for ele in dic_max:
        max_atoms+=dic_max[ele]

    print('Max. Number of sites',max_atoms)

    for configuracio in range(max_snapshot):
        lectura=fvxyz.readline()
        numAt=int(lectura)

        lcomment=fvxyz.readline()
        #print(l1,lcomment)
        fout.write(str(max_atoms)+'\n')
        fout.write(lcomment)
        snapshot=[]
        for atom in range(numAt):
            snapshot.append(fvxyz.readline())
        # Recalcul per cada atom´

        for ele_dic in dic_max:
            conta=0
            for lin in snapshot:
                words=lin.split()
                if words[0]==ele_dic:
                    fout.write(lin)
                    conta+=1
            # Afegir els sites necesaris fins al maxim
            for i in range(dic_max[ele_dic]-conta):
                fout.write(ele_dic+ '  -10.  -10.  -10.\n')

def convert_vxyz_old(nomfile,nom_out):
    fvxyz=open(nomfile)
    llista_num_atoms=[]
    n_max_HA=0
    n_max_A=0
    n_max_B=0
    n_max_N=0
    n_max_Na=0
    n_max_Cl=0
    n_max_HA2=0
    n_max_A2=0
    n_HA=0
    n_A=0
    n_B=0
    n_N = 0
    n_Na=0
    n_Cl=0
    n_HA2=0
    n_A2=0
    n_others=0
    for lin in fvxyz:
        words=lin.split()
        #print(lin[0])
        if lin[0].isnumeric()==True:  # Si és un número reiniciar comptadors
            llista_num_atoms.append(int(words[0]))       # El comentari no pot començar per un número
            n_max_HA=max(n_HA,n_max_HA)
            n_max_A=max(n_A,n_max_A)
            n_max_B=max(n_B,n_max_B)
            n_max_N = max(n_N,n_max_N)
            n_max_Na=max(n_Na,n_max_Na)
            n_max_Cl=max(n_Cl,n_max_Cl)
            n_max_HA2=max(n_HA2,n_max_HA2)
            n_max_A2=max(n_A2,n_max_A2)
            n_HA=0
            n_A=0
            n_B=0
            n_N=0
            n_Na=0
            n_Cl=0
            n_HA2=0
            n_A2=0
        elif words[0]=='HA':
            n_HA+=1
        elif words[0]=='A':
            n_A+=1
        elif words[0]=='B':
            n_B+=1
        elif words[0]=='N':
            n_N+=1
        elif words[0]=='Na':
            n_Na+=1
        elif words[0]=='Cl':
            n_Cl+=1
        elif words[0]=='HA2':
            n_HA2+=1
        elif words[0]=='A2':
            n_A2+=1
        else:
            n_others+=1

    print('Màxims:',n_max_HA,n_max_A,n_max_B,n_max_N,n_max_Cl,n_max_Na,n_max_HA2,n_max_A2)
    print('Parcials:',n_HA,n_A,n_B,n_N,n_Cl,n_Na,n_HA2,n_A2)
    print(llista_num_atoms)

    fvxyz.close()
    # Torna a obrir i generar el nou fitxer

    fvxyz=open(nomfile)
    fout=open(nom_out,'w')

    max_atoms=n_max_A+n_max_HA+n_max_B+n_max_N+n_max_Cl+n_max_Na+n_max_HA2+n_max_A2
    for configuracio in range(len(llista_num_atoms)):
        l1=fvxyz.readline()
        lcomment=fvxyz.readline()
        #print(l1,lcomment)
        fout.write(str(max_atoms)+'\n')
        fout.write(lcomment)
        snapshot=[]
        for atom in range(llista_num_atoms[configuracio]):
            snapshot.append(fvxyz.readline())
        # Recalcul per cada atom
        # Atom AH
        nl_AH=0
        for lin in snapshot:
            words=lin.split()
            if words[0]=='HA':
                #print(lin)
                fout.write(lin)
                nl_AH+=1
        for i in range(n_max_HA-nl_AH):
            fout.write('HA  -10.  -10.  -10.\n')
        # Atom A
        nl_A=0
        for lin in snapshot:
            words=lin.split()
            if words[0]=='A':
                #print(lin)
                fout.write(lin)
                nl_A+=1
        for i in range(n_max_A-nl_A):
            fout.write('A  -10.  -10.  -10.\n')
        # Atom B
        nl_B=0
        for lin in snapshot:
            words=lin.split()
            if words[0]=='B':
                #print(lin)
                fout.write(lin)
                nl_B+=1
        for i in range(n_max_B-nl_B):
            fout.write('B  -10.  -10.  -10.\n')
        nl_N=0
        for lin in snapshot:
            words=lin.split()
            if words[0]=='N':
                #print(lin)
                fout.write(lin)
                nl_N+=1
        for i in range(n_max_N-nl_N):
            fout.write('N  -10.  -10.  -10.\n')

        nl_Cl=0
        for lin in snapshot:
            words=lin.split()
            if words[0]=='Cl':
                #print(lin)
                fout.write(lin)
                nl_Cl+=1
        for i in range(n_max_Cl-nl_Cl):
            fout.write('Cl  -10.  -10.  -10.\n')
        nl_Na=0
        for lin in snapshot:
            words=lin.split()
            if words[0]=='Na':
                #print(lin)
                fout.write(lin)
                nl_Na+=1
        for i in range(n_max_Na-nl_Na):
            fout.write('Cl  -10.  -10.  -10.\n')
        nl_HA2=0
        for lin in snapshot:
            words=lin.split()
            if words[0]=='HA2':
                #print(lin)
                fout.write(lin)
                nl_HA2+=1
        for i in range(n_max_HA2-nl_HA2):
            fout.write('HA2   -10.  -10.  -10.\n')
        nl_A2=0
        for lin in snapshot:
            words=lin.split()
            if words[0]=='A2':
                #print(lin)
                fout.write(lin)
                nl_A2+=1
        for i in range(n_max_A2-nl_A2):
            fout.write('A2  -10.  -10.  -10.\n')





if __name__ == "__main__":
    print('In main')

#import tipus_particules as tp
#Temp=298  # K
#tp.calcul_gammas(Temp)
#tp.Llista_Tipus_Activa=[tp.TYPE_AH,tp.TYPE_A,tp.TYPE_Hp]
#tp.print_info_tipus_Actius()
