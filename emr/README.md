Install Ray on an EMR using a [bootstrap action](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-plan-bootstrap.html).

```
if grep isMaster /mnt/var/lib/info/instance.json | grep false; then
    sudo python3 -m pip install -U ray[all]==2.0.0

    RAY_HEAD_IP=$(grep "\"masterHost\":" /emr/instance-controller/lib/info/extraInstanceData.json | cut -f2 -d: | cut -f2 -d\")
...
```
