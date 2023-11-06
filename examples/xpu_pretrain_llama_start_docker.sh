#!/usr/bin/env bash
function device_config() {
    echo -e "${CYELLOW} XPU NUM: ${XPU_NUM} ${CEND}"
    XPU_NUM=$(ls /proc/xpu/ | grep dev | wc -l)
    DOCKER_DEVICE_CONFIG=" "
    if [ $XPU_NUM -gt 0 ]; then
        for ((idx=0; idx<=$XPU_NUM-1; idx++)); do
            DOCKER_DEVICE_CONFIG+=" --device=/dev/xpu${idx}:/dev/xpu${idx} "
        done
        DOCKER_DEVICE_CONFIG+=" --device=/dev/xpuctrl:/dev/xpuctrl "
    fi
}
CONTAINER_NAME="llama_dev"
version="v0.24"
device_config
docker run -dti --name "$CONTAINER_NAME" --privileged=true --net=host --security-opt=seccomp=unconfined \
                --cap-add=SYS_PTRACE --shm-size=32g \
                --restart=always \
                --ulimit=memlock=-1 --ulimit=nofile=120000 --ulimit=stack=67108864  \
                -v ~:/home -v `pwd`:/llama \
                $DOCKER_DEVICE_CONFIG iregistry.baidu-int.com/xmlir/xmlir_ubuntu_2004_x86_64:${version}
#               --cpuset-cpus=0-$((`nproc`-8)) \
# get xpu tool
docker exec -it $CONTAINER_NAME bash -c 'mkdir -p /opt/compiler/ && mkdir -p /usr/local/xpu/tools/' || true
docker cp ./gcc-8.2.tar.gz $CONTAINER_NAME:/opt/compiler/ || true
docker exec -it $CONTAINER_NAME bash -c 'cd /opt/compiler && tar -zxf /opt/compiler/gcc-8.2.tar.gz' || true
docker cp -L $(which xpu_smi) $CONTAINER_NAME:/bin/xpu_smi || true
docker exec -ti $CONTAINER_NAME bash
