# list all filenames in folder; exclude pytorch_components.py and pytorch_components_test.py
# loop thru all components in folder
    # check if they have __test__==True
    # check if they have __pytorch_component__==True
    # if so, grab the component that needs to be tested __to_be_tested__='ComponentName' and add to list
# use pytest.mark.parametrize to loop thru all components that need to be tested
    # use import_from_nested_path to grab ComponentConfig and Component
    # initialize ComponentConfig and Component with default values (shouldn't require any inputs)
    # grab Component.input_info and component.output_info
    # set Component.config.use_triton=False
    # create a tensor matching Component.input_info
    # run it through Component.forward()
    # assert it matches component.output_info
    # if Component.config.has_triton==True
        # set Component.config.use_triton=True
        # repeat the prior tensor creation, component running, and assertion (but now it's using Triton)
        # assert the output when use_triton==False and use_triton==True match a torch.allclose(abs=1e-2, rel=1e-1)