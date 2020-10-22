!pip install kfserving
from kubernetes import client
from kubernetes.client import V1Container

from kfserving import KFServingClient
from kfserving import constants
from kfserving import utils
from kfserving import V1alpha2EndpointSpec
from kfserving import V1alpha2PredictorSpec
from kfserving import V1alpha2InferenceServiceSpec
from kfserving import V1alpha2InferenceService
from kfserving import V1alpha2CustomSpec

namespace = utils.get_default_target_namespace()
api_version = constants.KFSERVING_GROUP + '/' + constants.KFSERVING_VERSION

default_endpoint_spec = V1alpha2EndpointSpec(
                          predictor=V1alpha2PredictorSpec(
                              custom=V1alpha2CustomSpec(
                                  container=V1Container(
                                      name="kfserving-model",
                                      image=f"abhishek1234docker/kfserving-model"))))

isvc = V1alpha2InferenceService(api_version=api_version,
                          kind=constants.KFSERVING_KIND,
                          metadata=client.V1ObjectMeta(
                              name='kfserving-model', namespace=namespace),
                          spec=V1alpha2InferenceServiceSpec(default=default_endpoint_spec))

KFServing = KFServingClient()
KFServing.create(isvc)

KFServing.get('kfserving-model', namespace=namespace, watch=True, timeout_seconds=120)