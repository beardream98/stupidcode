from datetime import datetime
from datetime import timedelta
from datetime import timezone
from tkinter import N
import os 
import wandb
def wandb_login():
    try:
        api_key = "ebe051612bfb733306f4e4b5df4b043050ebea6e"
        wandb.login(key=api_key)
        anony = None
    except:
        anony = "must"
        print('If you want to use your W&B account,  \nGet your W&B access token from here: https://wandb.ai/authorize')

def get_run(CONFIG,job_type,fold):

    if CONFIG["run_db"]==False or CONFIG["args"].local_rank!=0:
        return None
    HASH_NAME=job_type
    Group=os.getenv("WANDB_RUN_GROUP")
    if Group==None:
        Group="null"
    run = wandb.init(project='Query_classify', 
                     config=CONFIG,
                     job_type=job_type,
                     mode=os.getenv("UPLOAD_MODE"),
                     group=Group,
                     tags=[f"{CONFIG['model_name']}", f'{HASH_NAME}'],
                     name=f'{HASH_NAME}-fold-{fold}',
                     anonymous='must')
    return run

def wandb_utils(experimentName=None,upload=True):
    wandb_login()
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_now=""
    wandb.util.generate_id()
    if experimentName==None:
        experimentName="experiment-"
        beijing_now = utc_now.astimezone(SHA_TZ)

    os.environ["WANDB_RUN_GROUP"] = experimentName + str(beijing_now)[:24]
    if upload==False:
        os.environ["UPLOAD_MODE"]="offline"
    else:
        os.environ["UPLOAD_MODE"]="online"

        # os.environ['WANDB_MODE'] = 'dryrun'
        # os.environ['WANDB_MODE'] = 'run'

if __name__=='__main__':
    pass