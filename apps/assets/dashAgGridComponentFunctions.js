var dagcomponentfuncs = window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};

dagcomponentfuncs.DCC_GraphClickData = function (props) {
    const {setData} = props;
    function setProps() {
        const graphProps = arguments[0];
        if (graphProps['clickData']) {
            setData(graphProps);
        }
    }
    return React.createElement(window.dash_core_components.Graph, {
        figure: props.value,
        setProps,
        style: {height: '100%'},
        config: {displayModeBar: false},
    });
};

dagcomponentfuncs.DBC_Button_Simple = function (props) {
    const {setData, data} = props;

    function onClick() {
        setData();
    }
    return React.createElement(
        window.dash_bootstrap_components.Button,
        {
            onClick,
            color: props.color,
        },
        props.value
    );
};


//
//dagcomponentfuncs.DMC_Button = function (props) {
//    const {setData, data} = props;
//
//    function onClick() {
//        setData();
//    }
//    let leftIcon, rightIcon;
//    if (props.leftIcon) {
//        leftIcon = React.createElement(window.dash_iconify.DashIconify, {
//            icon: props.leftIcon,
//        });
//    }
//    if (props.rightIcon) {
//        rightIcon = React.createElement(window.dash_iconify.DashIconify, {
//            icon: props.rightIcon,
//        });
//    }
//    return React.createElement(
//        window.dash_mantine_components.Button,
//        {
//            onClick,
//            variant: props.variant,
//            color: props.color,
//            leftIcon,
//            rightIcon,
//            radius: props.radius,
//            style: {
//                margin: props.margin,
//                display: 'flex',
//                justifyContent: 'center',
//                alignItems: 'center',
//            },
//        },
//        props.value
//    );
//};
