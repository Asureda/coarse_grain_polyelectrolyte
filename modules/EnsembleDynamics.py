    def perform_sampling(self,npH):
        
        self.system.time = 0.
        start_time = time.time()
        self.num_As = np.zeros(self.n_samp_iter)
        self.num_As2 = np.zeros(self.n_samp_iter)
        self.num_B = np.zeros(self.n_samp_iter)
        self.qdist = np.zeros((self.n_samp_iter, self.n_mon))
        self.times = np.zeros(self.n_samp_iter)
        self.en_total = np.zeros(self.n_samp_iter)
        self.e_kin = np.zeros(self.n_samp_iter)
        self.e_coulomb = np.zeros(self.n_samp_iter)
        self.e_bonded = np.zeros(self.n_samp_iter)
        self.e_non_bonded = np.zeros(self.n_samp_iter)
        self.T_inst = np.zeros(self.n_samp_iter)
        self.rad_gyr = np.zeros(self.n_samp_iter)
        self.end_2end = np.zeros(self.n_samp_iter)

        polymer_list_sorted = np.arange(self.n_mon)
        for i in range(self.num_samples):
            if self.USE_WCA and np.random.random() < self.probability_reaction:
                self.system.integrator.run(self.number_integration_steps)
            if self.USE_WCA and np.random.random() < self.probability_reaction*(2.0/(self.n_mon-2)):
                self.system.integrator.run(self.number_integration_steps)

            self.reaction1.reaction(reaction_steps=self.n_nh + 1)
            self.reaction2.reaction(reaction_steps=self.n_nh2 + 1)

            if i%self.n_samp_iter ==0:
                j = (i // self.n_samp_iter)
                self.num_As[j] = self.system.number_of_particles(type=self.type_particles["N"])
                self.num_As2[j] = self.system.number_of_particles(type=self.type_particles["N2"])
                self.num_B[j] = self.system.number_of_particles(type=self.type_particles["B"])

                qn = self.system.part.by_ids(polymer_list_sorted).q
                self.qdist[j, :] = qn
                self.times[j] = self.system.time
                self.energy = self.system.analysis.energy()
                self.en_total[j] = self.energy['total']
                self.e_kin[j] = self.energy['kinetic']
                self.e_coulomb[j] = self.energy['coulomb']
                self.e_bonded[j] = self.energy["bonded"]
                self.e_non_bonded[j] = self.energy["non_bonded"]
                self.rad_gyr[j] = self.system.analysis.calc_rg(chain_start=0,number_of_chains=1,chain_length=self.n_mon)[0]
                self.end_2end[j] = self.system.analysis.calc_re(chain_start=0,number_of_chains=1,chain_length=self.n_mon)[0]
                self.T_inst[j] = 2. / 3. * self.e_kin[j] /len(self.system.part)

                if i%500 ==0:
                    print("time: {0:.2f}, n_N: {1:.2f}, n_N2: {2:.2f}".format(self.times[j], self.num_As[j], self.num_As2[j]))

                    print("time: {0:.2f}, energy: {1:.3f},e_kin: {2:.3f},e_coulomb: {3:.3f},e_bonded: {4:.3f},\
                    e_non_bonded: {5:.3f}, Temperature: {6:.3f}".format(self.times[j],self.en_total[j],self.e_kin[j],self.e_coulomb[j],self.e_bonded[j],self.e_non_bonded[j] \
                    ,self.T_inst[j]))

                    print("time: {0:.2f}, end-to-end: {1:.2f}, gyration radius: {2:.2f}".format(self.times[j], self.end_2end[j], self.rad_gyr[j]))

        print(" Completed. Time elapsed: {time.time()-start_time:.2f} seconds.")
