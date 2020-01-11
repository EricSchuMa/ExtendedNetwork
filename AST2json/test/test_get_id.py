from create_json_file import get_file_id


class TestAST():

    def test_get_file_id(self):
        file = "data/Havate/havate-openstack/proto-build/gui/horizon/Horizon_GUI/openstack_dashboard/dashboards/router/nexus1000v/tabs.py"
        print("The file id is: {}".format(get_file_id(file, True)))

