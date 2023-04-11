continue_path=os.path.join(CONFIG["ContinuePath"],f"LastCheckpoint{0}.pth")
checkpoint1=torch.load(continue_path)

modeldict=checkpoint["model_state_dict"]
modeldict1=checkpoint1["model_state_dict"]

for key in modeldict.keys():

    sum=(modeldict1[key].cpu().detach().numpy()\
        ==modeldict[key].cpu().detach().numpy()).sum()
    size=modeldict1[key].cpu().detach().numpy().size
    if size!=sum:
        print(key)

optdict=checkpoint["opt_state_dict"]
optdict1=checkpoint1["opt_state_dict"]

state=optdict["state"]
state1=optdict["state"]


for key in state.keys():
    for key1 in state[key].keys():
        if type(state[key][key1])!=torch.Tensor:
            if state[key][key1]!=state1[key][key1]:
                print(key,key1)
        else:
            sum=(state[key][key1].cpu().detach().numpy()\
            ==state1[key][key1].cpu().detach().numpy()).sum()
            size=state[key][key1].cpu().detach().numpy().size
            if size==sum:
                print(key,key1)