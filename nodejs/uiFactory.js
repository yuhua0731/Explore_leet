var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var UIFactory = /** @class */ (function () {
    function UIFactory() {
    }
    return UIFactory;
}());
var WinButton = /** @class */ (function () {
    function WinButton() {
    }
    WinButton.prototype.click = function () {
        console.log('show windows button');
    };
    return WinButton;
}());
var WinCheckBox = /** @class */ (function () {
    function WinCheckBox() {
    }
    WinCheckBox.prototype.checklist = function () {
        console.log('show windows checkbox');
    };
    return WinCheckBox;
}());
var WinDialog = /** @class */ (function () {
    function WinDialog() {
    }
    WinDialog.prototype.showModal = function () {
        console.log('show windows dialog');
    };
    return WinDialog;
}());
var Win_UIFactory = /** @class */ (function (_super) {
    __extends(Win_UIFactory, _super);
    function Win_UIFactory() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Win_UIFactory.prototype.createButton = function () {
        return new WinButton();
    };
    Win_UIFactory.prototype.createCheckbox = function () {
        return new WinCheckBox();
    };
    Win_UIFactory.prototype.createDialog = function () {
        return new WinDialog();
    };
    return Win_UIFactory;
}(UIFactory));
var LinuxButton = /** @class */ (function () {
    function LinuxButton() {
    }
    LinuxButton.prototype.click = function () {
        console.log('show linux button');
    };
    return LinuxButton;
}());
var LinuxCheckbox = /** @class */ (function () {
    function LinuxCheckbox() {
    }
    LinuxCheckbox.prototype.checklist = function () {
        console.log('show linux checkbox');
    };
    return LinuxCheckbox;
}());
var LinuxDialog = /** @class */ (function () {
    function LinuxDialog() {
    }
    LinuxDialog.prototype.showModal = function () {
        console.log('show linux dialog');
    };
    return LinuxDialog;
}());
var Linux_UIFactory = /** @class */ (function (_super) {
    __extends(Linux_UIFactory, _super);
    function Linux_UIFactory() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Linux_UIFactory.prototype.createButton = function () {
        return new LinuxButton();
    };
    Linux_UIFactory.prototype.createCheckbox = function () {
        return new LinuxCheckbox();
    };
    Linux_UIFactory.prototype.createDialog = function () {
        return new LinuxDialog();
    };
    return Linux_UIFactory;
}(UIFactory));
function app(ui) {
    ui.button.click();
    ui.dialog.showModal();
    ui.checkbox.checklist();
}
function mainApp() {
    var winUI = new Win_UIFactory();
    var button_win = winUI.createButton();
    var dialog_win = winUI.createDialog();
    var checkbox_win = winUI.createCheckbox();
    app({ button: button_win, dialog: dialog_win, checkbox: checkbox_win });
    var linuxUI = new Linux_UIFactory();
    var button_linux = linuxUI.createButton();
    var checkbox_linux = linuxUI.createCheckbox();
    var dialog_linux = linuxUI.createDialog();
    app({ button: button_linux, dialog: dialog_linux, checkbox: checkbox_linux });
}
mainApp();
