runber.py net=_pars/fncrc global_prob=[1.,-1.] fncrc_OUT
runber.py net=_pars/fncrc global_prob=[1.,-1.] fncrc_OUT
  790   python runber.py net=_pars/fncrc global_prob=[1.,1.] fncrc_R_OUT
  791   python runber.py net=_pars/fncrc global_prob=[1.,0.] fncrc_RR_OUT
  794   python runber.py net=_pars/fncrc use_existing=True mod_net=modf_net global_prob=[1.,-1.] fncrc_L_OUT
  795   python runber.py net=_pars/fncrc use_existing=True start=1 mult=2 mod_net=modf_net global_prob=[1.,-1.] fncrc_L_OUT
  796   python runber.py net=_pars/fncrc global_prob=[1.,1.] fncrc_R_OUT
  797   python runber.py net=_pars/fncrc use_existing=True start=1 mult=2 mod_net=modf_net global_prob=[1.,-1.] fncrc_L_OUT
  799   python runber.py net=_pars/fncrc use_existing=True start=1 mult=1 mod_net=modf_net global_prob=[1.,-1.] fncrc_L