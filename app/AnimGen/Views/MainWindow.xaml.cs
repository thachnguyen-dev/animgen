using System.Windows;
using AnimGen.ViewModels;

namespace AnimGen.Views;

public partial class MainWindow : Window
{
    public MainViewModel Vm { get; }

    public MainWindow(MainViewModel vm)
    {
        Vm = vm;
        DataContext = vm;
        InitializeComponent();
    }

    private async void AddWeapon_Click(object sender, RoutedEventArgs e)
    {
        if (Vm.Api == null) return;
        var dlg = new WeaponDialog(Vm.Api) { Owner = this };
        dlg.ShowDialog();
        try
        {
            await Vm.ReloadWeaponsAsync(dlg.SavedWeaponId);
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, Title, MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }
}
