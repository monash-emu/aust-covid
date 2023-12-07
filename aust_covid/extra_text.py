from emutools.tex import TexDoc
from inputs.constants import RUN_IDS, PRIMARY_ANALYSIS, BURN_IN, OPTI_DRAWS, ANALYSIS_FEATURES


def add_explanatory_blurb_to_tex(tex_doc: TexDoc):
    description = 'The following document describes the methods used in our analyses ' \
        'of the 2022 SARS-CoV-2 epidemic in Australia. ' \
        f'We constructed {len(RUN_IDS)} alternative dynamic transmission models ' \
        'based around the same core features. ' \
        f"These are named `{', '.join(RUN_IDS.keys())}' and were all based on the features " \
        'described in Sections \\ref{base_compartmental_structure}, \\ref{population}, ' \
        '\\ref{stratification}, \\ref{reinfection}, \\ref{mixing}. ' \
        f'Two of the models ({", ".join([analysis for analysis, feature in ANALYSIS_FEATURES.items() if feature["mob"]])}) ' \
        'incorporated additional structure to capture time-varying ' \
        'mobility \\ref{mobility_extension}, while two incorporated additional structure for time-varying ' \
        f'vaccination effects {", ".join([analysis for analysis, feature in ANALYSIS_FEATURES.items() if feature["vacc"]])} \\ref{{vaccination_extension}}, ' \
        'such that these additional features are applied factorially ' \
        'to the base model structure (therefore including one model with neither extension).\n\n' \
        'Each of the four alternative modelling approaches were then calibrated to the ' \
        'same target data for the 2022 Australian COVID-19 epidemic (see Section \\ref{targets}). ' \
        'The calibration algorithms were also harmonised to the greatest extent possible (see Section \\ref{calibration_methods}), ' \
        'although the two analysis approaches that included structure for vaccination ' \
        'required a different parameter to be substituted for the parameters used ' \
        'in the analyses not incorporating this structure (as described below). ' \
        'These four approaches were then compared with regards to their fit to ' \
        "the target data, with the `mob' analysis (with structure for mobility " \
        'but not for vaccination) found to achieve the highest likelihood. ' \
        f"As a result, this approach to analysis was selected as " \
        'the primary analysis (see Section \\ref{analysis_comparison})). ' \
        'This approach was used for the further analyses, ' \
        'including parameter inference (e.g. Section \\ref{calibration_results}). '
    tex_doc.add_line(description, 'Approach to analyses')


def add_model_structure_blurb_to_tex(tex_doc: TexDoc):
    description = 'This section describes the model features that were common to the model ' \
        'used for all four analyses. '
    tex_doc.add_line(description, 'Base compartmental structure')


def add_parameters_blurb_to_tex(tex_doc: TexDoc, cross_ref: bool=True):
    calib_ref = ' (See Section \\ref{calibrated_dispersion_parameters})' if cross_ref else ''
    prior_ref = 'Calibration priors are identified in the parameters table and illustrated in detail in Section \\ref{priors}. ' if cross_ref else ''
    description = 'All epidemiologial parameters, including those used in the calibration algorithm ' \
        'are presented in Table \\ref{params}. In addition to these epidemiological parameters, ' \
        'the dispersion parameter for the negative binomial distribution used when calculating ' \
        'the case time-series and death time-series contributions to the likelihood ' \
        f'were included in our calibration algorithm {calib_ref}. ' \
        'The approach to estimating the age-specific infection fatality rate for each ' \
        'modelled age group is described in \\ref{infection_fatality_rates}. ' \
        'All epidemiologically significant model parameters were included as priors ' \
        f'in our calibration algorithm.{prior_ref} '
    tex_doc.add_line(description, 'Parameters')


def add_likelihood_blurb_to_tex(tex_doc: TexDoc, cross_ref: bool=True, fig_ref: bool=True):
    target_ref = 'Section \\ref{targets})' if cross_ref else 'the Targets section'
    fig_ref = '\\ref{case_ranges}, \\ref{death_ranges} and \\ref{seropos_ranges}' if fig_ref else 'presented elsewhere'
    like_ref = ' (See Figure \\ref{like_comparison})' if cross_ref else ''
    description = 'We compared our four candidate analyses according to their goodness ' \
        f'of fit to the targets data (described under {target_ref}. ' \
        'The fit of all four of the models to the target data was considered adequate, ' \
        f"but the likelihood of the `{PRIMARY_ANALYSIS}' analysis was slightly higher than " \
        'that of the other three approaches, with the inclusion of the mobility structure ' \
        f"appearing to improve the calibration algorithm's fit to targets{like_ref}. " \
        f"For this reason, the `{PRIMARY_ANALYSIS}' analysis was considered as the primary analysis " \
        'throughout the remaining sections. ' \
        f'Figures {fig_ref} illustrate ' \
        'the fit of each candidate model to the target data for ' \
        'the notification, death and seropositive proportion respectively. '
    tex_doc.add_line(description, 'Analysis comparison')


def add_calibration_blurb_to_tex(tex_doc: TexDoc):
    description = "We calibrated our epidemiological model using the `DE Metropolis(Z)' algorithm " \
        'provided in the \\href{https://www.pymc.io/welcome.html}{PyMC} package for Bayesian inference.\n\n' \
        'Prior to running the algorithm, we ran a short optimisation using ' \
        "2-point differential evolution provided by Facebook Research's " \
        '\\href{https://facebookresearch.github.io/nevergrad/}{nevergrad} library. ' \
        'Initialisation of the optimisation algorithm was performed using Latin hypercube sampling ' \
        'to select initial parameter values that were broadly dispersed across the ' \
        f'multi-dimensional parameter space. The deliberately short optimisation algorithm of {OPTI_DRAWS} draws ' \
        'was used to move the parameter sets from these dispersed starting positions towards ' \
        'values with greater likelihood, but remained substantially dispersed from one-another. ' \
        '\n\nWe then ran the calibration algorithm from these starting points for 10,000 draws ' \
        'during the tuning phase, and for 50,000 draws during the calibration phase ' \
        f'of the algorithm, discarding the first {str(int(BURN_IN / 1000))},000 draws as burn-in. ' \
        'An identical and independent algorithm was applied for each of the four analysis approaches. '        
    tex_doc.add_line(description, 'Calibration methods')
    description = 'The metrics of the performance of our calibration algorithm are presented in ' \
        'Table \\ref{calibration_metrics}. '
    tex_doc.add_line(description, 'Calibration results', subsection='Calibration performance')
    description = 'Parameter-specific chain traces with parameter and chain-specific posterior densities ' \
        f"for the primary `{PRIMARY_ANALYSIS}' analysis " \
        'are presented in Figures \\ref{trace_fig_1}, \\ref{trace_fig_2} and \\ref{trace_fig_3}. ' \
        'These are used for the epidemiological interpretation of our results in the main manuscript. ' \
        'Overall posterior densitites (pooled over calibration chains) compared against prior distributions are ' \
        'presented in Figures \\ref{comp_fig_1} and \\ref{comp_fig_2}. '
    tex_doc.add_line(description, 'Calibration results', subsection='Parameter inference')


def add_dispersion_blurb_to_tex(tex_doc: TexDoc):
    description = 'Figure \\ref{dispersion_examples} provides an illustration of the effect of specific values of the calibrated dispersion ' \
        'parameter used in the calibration algorithm to adjust the calculation of the contribution ' \
        'to the likelihood from the notifications and deaths time series. '
    tex_doc.add_line(description, 'Targets', subsection='Calibrated dispersion parameters')


def add_mobility_blurb_to_tex(tex_doc: TexDoc):
    description = 'The two scaling functions developed in the previous section were used to adjust ' \
        'rates of contact over time in the two time-varying matrix locations. ' \
        'These were summed with the two static locations to obtain the final matrix. ' \
        'Examples of the final effect of the matrix scaling function on the dynamic ' \
        'mixing matrices are presented in Figure \\ref{example_matrices}. '
    tex_doc.add_line(description, 'Mobility extension', subsection='Application')


def add_vaccination_blurb_to_tex(tex_doc: TexDoc):
    description = "Although Australia's population was relatively unexposed to SARS-CoV-2 " \
        'infection and so had little natural immunity at the start of our simulation period, ' \
        'the population had extensive vaccination-derived immunity across eligible age groups. ' \
        'This is illustrated in Figure \\ref{full_vacc}, which shows that most age groups ' \
        'had reached very high coverage with a second-dose of vaccine by early 2022. ' \
        'As such, we considered that the continuing roll-out of second doses were ' \
        'very unlikely to have substantially modified the epidemic, ' \
        'particularly given the questionable impact of such vaccination programs on ' \
        'onward transmission.\n\n' \
        'We therefore considered programs that were rolled out over the course of 2022 ' \
        'for their potential impact on transmission. ' \
        'As shown in Figure \\ref{program_coverage}, we considered that the most likely ' \
        'programs to have had an effect on transmission through 2022 were the fourth dose ' \
        "`winter booster' program, and the primary course (completing second doses) " \
        'program for children aged 5 to 11 years. ' \
        'The heterogeneous immunity stratification of the base model was utilised to ' \
        'consider the impact of these programs on community transmission, ' \
        'as described in the following section. '
    tex_doc.add_line(description, 'Vaccination extension', subsection='Rationale')
    description = 'Using the various reporting streams from which the vaccination data ' \
        'were derived (illustrated as different colours in Figure \\ref{program_coverage}), ' \
        'we calculated the total number of persons vaccinated. ' \
        'Next, we converted this to a proportion using the population denominators supplied ' \
        'by the Commonwealth in the same dataset. ' \
        'We then calculated the proportion of the population previously unvaccinated ' \
        'under each of these programs and divided by the time interval over which this occurred ' \
        'to calculate the rate at which these population groups ' \
        'received vaccination (Figure \\ref{vacc_implement}). ' \
        'These were then applied as unidirectional flows taht transitioned persons from ' \
        'the unvaccinated stratum to the vaccinated stratum. ' \
        'For this extended model configuration, a third stratum was added to ' \
        'the immunity stratification. ' \
        'The reported coverage and coverage lagged by 14 days are compared against the ' \
        'modelled population distribution across the three immunity strata in Figure \\ref{vaccination_distribution}. '
    tex_doc.add_line(description, 'Vaccination extension', subsection='Application')


def add_abstract_blurb_to_tex(tex_doc: TexDoc):
    description = 'Software engineering is advancing at astonishing speed, ' \
        'with packages now available to support many stages of data science pipelines. ' \
        'These packages can support infectious disease modelling to be more robust, ' \
        'efficient and transparent, which has been particularly important during the COVID-19 pandemic. ' \
        'We developed a package for the construction of infectious disease models, ' \
        'integrated this with several open-source libraries and applied this pipeline ' \
        "to multiple data sources relevant to Australia's 2022 COVID-19 epidemic. " \
        'Extending the base model to include mobility effects slightly improved model fit, ' \
        'but including the effect of 2022 vaccination programs on transmission did not. ' \
        'Our simulations suggested that one in every two to six COVID-19 episodes were detected, ' \
        'subsequently emerging Omicron subvariants escaped 30 to 60\% of recently acquired natural immunity ' \
        'and that natural immunity lasted only one to eight months. ' \
        'We documented our analyses algorithmically and present our methods in conjunction with interactive online notebooks.'
    tex_doc.add_line(description, 'Abstract')


def add_summary_to_tex(tex_doc: TexDoc):
    description = 'This manuscript outlines a novel approach to the development of mathematical models of infectious diseases, ' \
        'applied to understand the epidemiology of COVID-19 in Australia in 2022.'
    tex_doc.add_line(description, 'One sentence summary')


def add_intro_blurbs_to_tex(tex_doc: TexDoc):
    description = 'Throughout the pandemic, epidemiological modelling has been used to influence ' \
        'some of the most significant and intrusive public health policy decisions in history, ' \
        'providing analyses to justify a range of programs that extended from lockdowns to vaccination.\\cite{mccaw2022,pagel2022} ' \
        'However, this policy impact brings with it a responsibility for modellers to ensure results are accurate, '  \
        'transparent and effectively communicated not only to policy makers, ' \
        'but also the public who are impacted by such decisions.\\cite{pagel2022}\n'
    tex_doc.add_line(description, 'Introduction')

    description = 'While libraries are available to support many components of the data science pipeline, ' \
        'the application of software engineering principles to epidemiological modelling remains limited.\\cite{horner2020} ' \
        'The rapid growth of data science as a field, and the corresponding investment in ' \
        'platform development provides constant opportunities to expand the range of packages ' \
        'that can be integrated with such models, provided the model code itself is developed with this in mind. ' \
        'In particular, a software package whose single responsibility is model construction can separate ' \
        'this concern from the multiple other stages in the formulation of a modelling-based analysis.'
    tex_doc.add_line(description, 'Introduction')

    description = "Australia's 2022 epidemic provides an important case study for understanding " \
        'the epidemiological characteristics of COVID-19, because of the distinct epidemic waves, ' \
        'negligible prevalence of natural immunity from past infection,\\cite{machalek2022,vette2022,gidding2021} ' \
        'stable vaccination coverage and multiple high-quality data streams. ' \
        'Australia pursued an elimination approach through the first two years of the pandemic, ' \
        'achieving one of the lowest COVID-19-related mortality rates in the world.\\cite{wang2022} ' \
        'Shortly after achieving very high coverage of wild-type vaccination, ' \
        'most of the country emerged abruptly from this elimination phase in 2022, ' \
        'relaxing most restrictions on population mobility as the Omicron variant rapidly replaced ' \
        'the preceding Delta SARS-CoV-2 variant of concern.\\cite{covidepireport2022} ' \
        'Through 2022, national data are available that include a daily time-series for cases and deaths, ' \
        'serial survey-derived testing behaviour, population mobility, vaccination coverage and seroprevalence. ' \
        'Of particular value, the serial serosurveys demonstrate a rapid rise in nucleocapsid antibodies to ' \
        'more than 65\% seroprevalence by late August 2022.\\cite{machalek2023} The combination of these data sources provides ' \
        'the opportunity to improve our understanding of COVID-19 epidemiology in the Omicron era, ' \
        'such as quantification of the case detection rate and characteristics of population immunity. '
    tex_doc.add_line(description, 'Introduction')

    description = 'We developed a suite of software tools to support evidence-based policies, ' \
        'which was applied to several countries of the Asia-Pacific Region in collaboration with ' \
        'the World Health Organization and other regional public health agencies.' \
        '\\cite{trauer2021a,caldwell2021a,jayasundara2021,hughes2022}' \
        'Our platform is based around a library for the construction of compartmental models of ' \
        'infectious disease transmission and is now integrated with publicly available libraries for numerical computing, ' \
        'optimisation, Bayesian inference, data visualisation and scientific documentation.' \
        '\\cite{salvatier2016,rapin2018,kumar2019,plotly2015} ' \
        'The combined platform constitutes an end-to-end pipeline for infectious disease modelling, ' \
        'which we used to derive insights into COVID-19 epidemiology through its application to ' \
        "Australia's three-wave 2022 epidemic."
    tex_doc.add_line(description, 'Introduction')


def add_methods_blurbs_to_tex(tex_doc: TexDoc):
    description = 'We constructed four candidate models with common underlying characteristics ' \
        'to represent COVID-19 dynamics during the course of 2022 in Australia. ' \
        'Epidemiological details are presented in the repository for this analysis ' \
        'and described in detail in our Supplemental Material, ' \
        'which is algorithmically generated from the code used in model construction ' \
        'to ensure accuracy of documentation. ' \
        'In brief, we built an SEIRS model with chained serial latent and infectious compartments ' \
        'and reinfection from the waned (second S) compartment. ' \
        'To this we added age structure in five-year bands from 0-4 years to 75 years and above. ' \
        'Age stratification determined the initial population distribution, ' \
        'and an age-specific mixing matrix was adapted to the Australian population structure ' \
        'from United Kingdom survey data was applied to capture heterogeneous mixing between age groups.\\cite{mossong2008} ' \
        'The model was further stratified into Western Australia (WA) ' \
        'and the other major jurisdictions of Australia to acknowledge the negligible community transmission ' \
        'in WA prior to the re-opening of internal borders to the state. ' \
        'Further stratification was applied to replicate all model compartments ' \
        'into two populations with differing levels of immunity to infection and reinfection, ' \
        'with no transition between these two classes permitted in the base model configuration ' \
        '(without vaccination extension). Three subvariant strains were introduced into the base model ' \
        'through the course of the simulation to represent the BA.1, BA.2 and BA.5 subvariants of Omicron, ' \
        'with incomplete cross-immunity to subsequent strains during the early recovered stage ' \
        'conferred through infection with earlier strains.\n'
    tex_doc.add_line(description, 'Materials and methods')

    description = 'The unextended model described in the preceding paragraph was elaborated in two respects. ' \
        'First, the mixing matrix that remains fixed over modelled time in the unextended model ' \
        'was allowed to vary over time, with the location-specific contribution ' \
        'to each cell of the matrix scaled according to metrics sourced from ' \
        "Google's Community Mobility Reports.\\cite{googlemob2023}\n"
    tex_doc.add_line(description, 'Materials and methods')

    description = 'Second, the model was extended to allow that the historical profile of ' \
        'vaccination through 2022 could have influenced rates of infection. ' \
        "Under this alternative analysis, all the model's initial population " \
        'was assigned to the non-immune category, with population then transitioning ' \
        'to the partially immune class as new vaccination programs (booster and paediatric) ' \
        'were rolled out through 2022. Vaccine-derived immunity was then allowed to wane, ' \
        'with vaccinated persons returning to a third immunity stratum ' \
        'with the same susceptibility to infection as those who had never received vaccination ' \
        'under these programs.\n'
    tex_doc.add_line(description, 'Materials and methods')

    description = 'From these two extensions to the base model, ' \
        'we created four alternative analytical approaches: ' \
        'no additional structure (``none"), mobility extension only (``mob"), ' \
        'vaccination extension only (``vacc"), and both mobility and vaccination extensions (``both").\n'
    tex_doc.add_line(description, 'Materials and methods')

    description = 'Last, we calibrated each of the four candidate models described ' \
        'in the preceding paragraph to publicly available data ' \
        'for three empirical indicators of the COVID-19 epidemic through 2022: ' \
        'the seven-day moving average of national daily time-series for case notifications, ' \
        'the seven-day moving average of national daily time-series for deaths ' \
        'and the results of a nationally representative adult blood donor seroprevalence survey ' \
        'at three key time points in 2022. Equivalent modelled quantities ' \
        'to notifications and deaths were estimated through a convolution approach ' \
        'that allowed a gamma-distributed delay from onset of symptoms ' \
        '(taken as the time of transition from the first to the second serial infectious compartment) ' \
        'to notification or death. The proportion of the population no longer in ' \
        'the original susceptible compartment was further compared to seroprevalence estimates of ' \
        'SARS-CoV-2 exposure, which were adjusted for nucleocapsid test sensitivity and ' \
        'lagged forward by 14 days. Model calibration was then achieved independently for ' \
        'each of the four candidate models using the PyMC implementation of ' \
        'the differential evolution Metropolis algorithm ``DEMetropolis(Z)".\\cite{salvatier2016} ' \
        'All important epidemiological parameter inputs were included in the calibration algorithm, ' \
        'creating a 17-dimensional parameter space for exploration. ' \
        'The approach with only mobility implemented was selected as the primary analysis for parameter inference, ' \
        'largely because of its superior fit to seroprevalence estimates.\n'
    tex_doc.add_line(description, 'Materials and methods')


def add_results_blurbs_to_tex(tex_doc: TexDoc):
    description = 'We released a suite of open-source packages to support infectious disease modelling ' \
        'and used these packages to represent the key epidemiological processes relevant to ' \
        "Australia's 2022 COVID-19 epidemic (Figure \\ref{aust_covid_code_structure}). At the heart of our pipeline, " \
        'we developed the summer Python package to support easy and reliable construction of ' \
        'infectious disease models through an epidemiologically intuitive application programming interface. ' \
        "summer's backend is integrated with Google's jax library for high-performance numerical computing. " \
        'We validated summer against a popular textbook of infectious diseases modelling,\\cite{vynnycky2010} ' \
        'demonstrating that it could recover the behaviours of a wide range of infectious diseases models ' \
        'through a series of \\href{https://github.com/monash-emu/summer-vynnycky-white-validation}{jupyter notebooks}. ' \
        'Next, through a series of interactive Google Colab-hosted ' \
        '\\href{https://github.com/monash-emu/summer-textbook}{notebooks}, ' \
        'we released a textbook of infectious disease modelling that systematically demonstrates ' \
        'core infectious disease modelling principles using summer. ' \
        'Last, we released estival, a wrapper for the calibration and optimisation of summer-based models, ' \
        'which supports the integration of these models with current ' \
        'state-of-the-art calibration, optimisation and interpretation platforms, ' \
        "including the PyMC package for Bayesian inference\\cite{salvatier2016} and Facebook's nevergrad library for " \
        'gradient-free optimisation.\\cite{rapin2018} ArviZ was used for Bayesian diagnostics,\\cite{kumar2019} ' \
        'with interactive visuals produced using plotly.\\cite{plotly2015}\n'
    tex_doc.add_line(description, 'Results')

    description = 'The analysis is provided as an installable Python package that incorporates ' \
        'interactive Google Colab Jupyter notebooks for the inspection of model features and interrogation of outputs. ' \
        'Through this approach, we fit a complex model of COVID-19 dynamics ' \
        '(1984 compartments under the base case, 2976 compartments under the vaccination extension) ' \
        'to three key epidemiological indicators (cases, deaths and nucleocapsid antibody seroprevalence, ' \
        'a marker of having ever being infected).'
    tex_doc.add_line(description, 'Results')

    description = 'We considered four candidate models with and without extended structure for ' \
        'mobility and vaccination for their ability to capture the broad epidemic profile of ' \
        "Australia's 2022 Omicron waves (i.e. mobility extension, vaccination extension, " \
        'both extensions, and neither). All four models were able to capture ' \
        'the broad epidemic profile we targeted, with each achieving a good fit to the time-series of deaths. ' \
        'The model configurations with additional structure for scaling contact rates ' \
        'with mobility data achieved a somewhat better fit to the seroprevalence targets than ' \
        'the two configurations without this extension ' \
        '(median seroprevalence likelihood contribution 0.6 versus 0.9, Figure \\ref{like_comparison}). ' \
        'Inclusion of additional model structure for time-varying vaccination-related immunity to ' \
        'infection resulted in a slightly poorer fit to the time-series for cases ' \
        '(median cases likelihood contribution -12.7 versus -12.2 to -12.4 for the other analyses). ' \
        'We therefore selected the mobility extension model as the primary model analysis for ' \
        'consideration in the following sections. Additional approaches to model construction, ' \
        'calibration and interpretation of results can easily be explored via ' \
        'the interactive notebooks available on the project \\href{https://github.com/monash-emu/aust-covid}{homepage}.'
    tex_doc.add_line(description, 'Results', subsection='Candidate model comparison')

    description = "Leveraging Google's jax package and calibration algorithms from PyMC, " \
        'epidemiological models of 1984 to 2976 compartments ' \
        '(depending on application of the vaccination extension) ' \
        'completed 60,000 iterations per chain within 4-8 hours on 8-core 3rd Generation Intel Xeon machines ' \
        'clocked at 2.9 to 3.5 GHz. For the primary (mobility extension) analysis, ' \
        'each core completed 3.62 model iterations per second. ' \
        'Metrics of the calibration algorithm for the primary analysis are presented in the Supplemental Material. ' \
        'The algorithm achieved highly satisfying chain convergence, ' \
        'with the $\\hat{R}$ statistic for all parameters below 1.05 and all effective sample sizes above 150 ' \
        '(Supplemental Figures \\ref{trace_fig_1} \\ref{trace_fig_2} \\ref{trace_fig_3}).\n'
    tex_doc.add_line(description, 'Results', subsection='Calibration results')

    description = 'Figure \\ref{base_ranges} shows model fit for each of the target epidemiological indicators. ' \
        'Calibration fit was better for deaths than for case notifications, ' \
        'which is reflected in the markedly lower likelihood contributions for the cases calibration target ' \
        '(median -12.2) than for the deaths targets (median -4.7) (Figure \\ref{like_comparison}). ' \
        'This difference was particularly noticeable during the first (BA.1) wave of 2022, ' \
        'at a time when notifications may have been a more variable epidemic indicator ' \
        'as Australia struggled to scale testing capacity up to match demand (Figure \\ref{base_ranges}).\\cite{timms2022} ' \
        'Under the constraint that BA.1 and BA.5 had the same modelled severity,\\cite{wolter2022} ' \
        'accepted model runs often under-estimated the peak number of deaths for the third (BA.5) wave of 2022. ' \
        'Our results typically showed a higher seroprevalence than estimated from the serosurvey target values ' \
        'for its first round, but lower for the third. ' \
        'The relative contribution of each variant and each infection process ' \
        '(i.e. de novo infection, early reinfection due to immune escape and late reinfection due to waned immunity) ' \
        'is presented in Figure \\ref{infect_process}.'
    tex_doc.add_line(description, 'Results', subsection='Calibration results')

    description = 'The short prior estimates for the durations for the latent and infectious periods\\cite{xin2022,wu2023} ' \
        'were not substantially influenced by the process of fitting to target data, ' \
        'while the period of partial immunity following infection was estimated to be markedly shorter than ' \
        'our prior belief (Figure \\ref{comp_fig}). Our uninformative prior estimate for the proportion of cases detected was ' \
        'also substantially influenced by the fitting process, ' \
        'suggesting that the highest case detection early in the BA.1 epidemic wave was around 17 to 50\%. ' \
        'The infection fatality rate had to be inflated around two- to three-fold from ' \
        'the baseline estimates taken from a population-wide epidemiological study in Denmark.\\cite{erikstrup2022} ' \
        'The extent of immune escape of both BA.2 and BA.5 against previous infection with ' \
        'other sub-variant strains was moderately greater than anticipated in our prior distributions,25-30 ' \
        '\\cite{altarawneh2022b,carazo2023,chemaitelly2022a,malato2022b,powell2023,tan2023}' \
        'and was centred around a value of 50\% for both subvariants ' \
        '(i.e. past infection only protected half as much against early reinfection with ' \
        'novel subvariants as compared to against early reinfection with the previously infecting subvariant). ' \
        'The relative reduction in severity of BA.2 compared to BA.1 and BA.5 was consistent with past evidence,' \
        '\\cite{wolter2022,hansen2023,robertson2023,webster2022} ' \
        'while little additional information was obtained for ' \
        'the time to WA fully mixing with the rest of the country ' \
        'or the parameters pertaining to the convolution processes for notifications and deaths. ' \
        'The seeding time parameters resulted in epidemic profiles that were consistent with ' \
        'reports of national genomic data (see Supplemental Figure \\ref{variant_prop_spaghetti}).\\cite{covidepireport2023} ' \
        'On examination of the bivariate distributions of combinations of two parameters, ' \
        'the contact rate parameter showed expected inverse interactions with the extent of population immunity ' \
        'and the infectious duration parameters (Supplemental Figure \\ref{key_param_correlation}). ' \
        'The association of shorter duration of post-infection immunity and a lower case detection proportion ' \
        'can be attributed to both these processes being associated with larger modelled epidemics. ' \
        'The expected association between short duration of post-infection immunity and ' \
        'the extent of BA.2 and BA.5 escape was observed, but was modest.'
    tex_doc.add_line(description, 'Results', subsection='Parameter inference')


def add_discussion_blurbs_to_tex(tex_doc: TexDoc):
    description = 'Our analysis demonstrates the feasibility of integrating epidemiological modelling with ' \
        'advanced tools in data science, including open-source tools produced by user communities and Big Tech. ' \
        'Our results supported a major role for Omicron subvariants in driving the three major epidemic waves observed ' \
        'in Australia over the course of 2022. With each subvariant epidemic following closely on from the preceding wave, ' \
        'most accepted model configurations required a combination of both high levels of immune escape, ' \
        'as well as a short period of natural immunity. ' \
        'In addition to improving efficiency and expanding the functional possibilities from our platform, ' \
        'our methodological approach reduces the volume of code that the modeller must produce, ' \
        'thereby enhancing transparency and minimising opportunities for errors.\n'
    tex_doc.add_line(description, 'Discussion')

    description = 'Of our candidate models, all were able to find epidemiologically plausible parameter space ' \
        'and were generally associated with similar results in relation to parameter inference. ' \
        'The coherence of our epidemiological model with multiple data sources simultaneously permitted our analysis to ' \
        'capture a range of model trajectories and associated parameter sets that could accurately represent ' \
        "Australia's 2022 COVID-19 epidemics. We achieved a closer fit to " \
        'the time-series of deaths than we achieved for cases, ' \
        'which may be attributable to the higher quality or greater consistency of the mortality data during ' \
        'a period when testing recommendations and test availability changed markedly. ' \
        'By contrast to our expectations in late 2021 when the Omicron variant first emerged in Southern Africa,\\cite{viana2022} ' \
        "the first (BA.1) epidemic wave of 2022 was likely not Australia's largest, " \
        'with the subsequent waves (largely attributable to BA.2 and BA.5) also associated with substantial attack rates. ' \
        'The vaccination extension did not improve calibration metrics, ' \
        'which may be attributable to the complex evolving profile of vaccination-induced population immunity, ' \
        'or the limited indirect protection conferred by vaccination.\\cite{braeye2023} ' \
        'By contrast, population mobility appeared to improve model fit slightly, ' \
        'possibly by allowing for a smaller initial BA.1 wave through capturing the ``shadow lockdown" in summer 2021/2022.' \
        '\\cite{dunstan2022,lloyd2022}\n'
    tex_doc.add_line(description, 'Discussion')

    description = 'Parameter posterior estimates suggested a greater level of immune escape and ' \
        'a shorter duration of immunity than our prior beliefs from the literature,' \
        '\\cite{edridge2020,kissler2020,townsend2021} ' \
        "which can be attributed to the rapid succession of each subvariant's wave as " \
        'the preceding wave had only recently begun to decline. ' \
        'Our inflation factor for the infection fatality rate suggested considerably greater severity than ' \
        'observed in a population with a longer history of COVID-19 epidemics,\\cite{erikstrup2022} ' \
        'which may relate to lower natural immunity to COVID-19 along with a larger vulnerable population after ' \
        'two years of lower circulation of influenza and other viruses.\\cite{oneill2023}\n'
    tex_doc.add_line(description, 'Discussion')

    description = 'The availability of seroprevalence estimates for nucleocapsid antibodies that increased markedly from ' \
        'close to zero to nearly 80\% within the simulation window we considered markedly increases ' \
        'our confidence that our analysis captures the overall epidemic size.\\cite{machalek2023} ' \
        'Although we did not consider antibody waning, ' \
        'the greatest delay from infection to antibody measurement relevant to our analysis would have been ' \
        'infection early in the BA.1 wave (January 2022) followed by antibody measurement in August 2022 ' \
        '(for persons not infected in the intervening waves), ' \
        'whereas these antibodies are known to be well maintained over ten months.\\cite{vanelslande2022} ' \
        "Moreover, given the apparently large size of Australia's first BA.1 wave, " \
        'we found it surprising that adult seroprevalence only reached 20.7\% after the first wave, ' \
        'whereas the subsequent estimates implied around 50\% of persons not previously infected ' \
        'were exposed within the subsequent windows between survey rounds (February to June and June to August). ' \
        'There are several possible causes of bias in such estimates, ' \
        'including selection bias of blood donors by comparison to the general population,\\cite{burgdoff2017} ' \
        'which has not been quantified in the Australian context to our knowledge. ' \
        'Rather than attempt to adjust for such biases, ' \
        'we acknowledged the bidirectional uncertainty in these data through the likelihood calculation for ' \
        'the seroprevalence calibration target.\n'
    tex_doc.add_line(description, 'Discussion')

    description = 'More generally, while adjusting our approach to allow for a smaller first wave would have ' \
        'improved fit to the first seroprevalence estimate ' \
        '(e.g. through a higher initial case detection ratio), ' \
        'a lower absolute epidemic peak would also have led to a flatter epidemic peak. ' \
        'By contrast, the first (BA.1 wave) notifications peak was very sharp in shape. ' \
        'Although the shape of this wave was likely modified by changing reporting requirements and ' \
        'poor national availability of rapid antigen tests, this implies a large epidemic. ' \
        'The mobility extension models we considered improved fit to the seroprevalence estimates, ' \
        'implying that the ``shadow" (public-led) lockdown over the 2021-2022 summer was ' \
        'likely part of the explanation for this wave being smaller than expected.\\cite{dunstan2022,lloyd2022} ' \
        'We considered that accepting model runs that modestly over-estimated ' \
        'the first seroprevalence estimate but under-estimated the latter two estimates was the optimal balance, ' \
        'as was the case with our chosen primary analysis for parameter inference.\n'
    tex_doc.add_line(description, 'Discussion')

    description = 'By contrast to mobility, allowing for the additional vaccination programs rolled out during 2022 to ' \
        'modify transmission after Australia had reached close to complete coverage with ' \
        'the two-dose primary courses did not improve model fit. ' \
        'The greatest modelled effect of these programs would be the roll-out of ' \
        'the third dose program for persons aged 16 and above, ' \
        'which reached its greatest rate in February and March 2022 ' \
        '(with all vaccine-related effects lagged to 14 days later). ' \
        'During this period, the modelled effective reproduction number had begun to increase following a nadir, ' \
        'by contrast to its marked decline during January driven by natural immunity following the major BA.1 wave. ' \
        'Therefore, the timing of this program did not help to explain the epidemic profile over a period when ' \
        'the overall change in population immunity was difficult to determine because of ' \
        'the simultaneous implementation of several vaccination programs when immunity from past vaccination was waning.\n'
    tex_doc.add_line(description, 'Discussion')

    description = 'Our final model structure aimed to balance parsimony against complexity and ' \
        'avoid over-fitting to the available data. Our analysis points to several features that may improve model fit. ' \
        'In addition to the considerations around seroprevalence targets and epidemic size discussed above, ' \
        'we did not allow for a direct effect of vaccination on severe outcomes, ' \
        'for the BA.5 subvariant to be more severe than both BA.1 and BA.2, ' \
        'or for population heterogeneity in spatial structure or immunity beyond our two-category approach. ' \
        'Such configurations and others of interest to the reader can be explored through our interactive notebooks.'
    tex_doc.add_line(description, 'Discussion')


def add_conclusions_blurb_to_tex(tex_doc: TexDoc):
    description = "Australia's 2022 COVID-19 epidemic was characterised by overlapping waves that " \
        'were driven by Omicron subvariants that exhibited substantial immune-escape properties, ' \
        'with rapidly waning immunity to past infection also likely contributing. ' \
        'The multiple data sources and clear swings in the epidemic profile and reproduction number allowed ' \
        'our beliefs pertaining to several epidemiological parameters to be substantially updated, ' \
        'suggesting a peak case detection rate of 17 to 50\% and higher infection fatality rates than ' \
        'observed in a setting with a greater experience with COVID-19. ' \
        'Our pipeline for infectious disease modelling supports greater understanding of ' \
        'these results through interactive notebooks and online visuals and could constitute ' \
        'a new paradigm for infectious disease modelling.'
    tex_doc.add_line(description, 'Conclusions')


def add_acknowledgements_blurb_to_tex(tex_doc: TexDoc):
    description = 'James Trauer is supported by a Discovery Early Career Researcher Award from the Australian Research Council (DE230100730). ' \
        'The Epidemiological Modelling Unit at the School of Public Health and Preventive Medicine (EMU) ' \
        'was supported by a Rapid Research Digital Infrastructure COVID-19 grant from ' \
        'the Medical Research Future Fund during 2021 and 2022 (RRDHI000027). ' \
        'EMU also provided modelling to countries of the Asia-Pacific through ' \
        'a series of contracts with the World Health Organization Western Pacific and South East Asia Regional Offices over the course of the pandemic, ' \
        'through which much of the software development underpinning this analysis was undertaken.'
    tex_doc.add_line(description, 'Acknowledgements', 'Funding')

    description = 'Conceptualization: JMT\n' \
        'Data curation: JMT, AEH, DSS\n' \
        'Formal analysis: JMT\n' \
        'Funding acquisition: JMT\n' \
        'Investigation: JMT, AEH\n' \
        'Methodology: JMT, AEH, DSS, MTM, ASH, ESM, RR\n' \
        'Software: JMT, DS\n' \
        'Supervision: RR\n' \
        'Visualization: JMT\n' \
        'Writing original draft: JMT\n' \
        'Writing - review & editing: JMT, AEH, DSS, MTM, ASH, ESM, RR'
    tex_doc.add_line(description, 'Acknowledgements', 'Author contributions')

    description = 'The authors declare no competing interests.'
    tex_doc.add_line(description, 'Acknowledgements', 'Competing interests')

    description = 'All data sources used for the calibration and parameterisation of models is obtained from open-sources or published literature. ' \
        'All sources for external data are documented in the main manuscript, methods or supplementary materials. ' \
        'All code for this analysis is available at the analysis \\href{https://github.com/monash-emu/aust-covid}{homepage}. ' \
        'A number of Google Colab notebooks that allow anyone to interact with the model code, ' \
        'explore calibration results and outputs and test model assumptions are available at the provided github link. '
    tex_doc.add_line(description, 'Acknowledgements', 'Data and materials availability')
